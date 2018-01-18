"""Unit tests for pipelines expressed via dask.bag


"""

import logging
import os
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from dask import bag

from arl.data.polarisation import PolarisationFrame

from arl.image.operations import export_image_to_fits, smooth_image, qa_image
from arl.imaging.base import create_image_from_visibility, predict_skycomponent_visibility, \
    predict_skycomponent_visibility
from arl.skycomponent.operations import insert_skycomponent
from arl.util.testing_support import create_named_configuration, ingest_unittest_visibility, \
    create_unittest_components, insert_unittest_errors

from arl.graphs.bags import reify
from arl.pipelines.bags import continuum_imaging_pipeline_bag, ical_pipeline_bag

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestPipelinesBags(unittest.TestCase):
    def setUp(self):
        
        self.compute = True
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        self.params = {'npixel': 512,
                       'nchan': 1,
                       'reffrequency': 1e8,
                       'facets': 1,
                       'padding': 2,
                       'oversampling': 2,
                       'kernel': '2d',
                       'wstep': 4.0,
                       'vis_slices': 1,
                       'wstack': None,
                       'timeslice': None}
    
    def actualSetUp(self, add_errors=False, freqwin=7, block=False, dospectral=True, dopol=False):
        cellsize = 0.001
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = freqwin
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        if freqwin > 1:
            self.channelwidth = numpy.array(freqwin * [self.frequency[1] - self.frequency[0]])
        else:
            self.channelwidth = numpy.array([1e6])
        
        if dopol:
            self.vis_pol = PolarisationFrame('linear')
            self.image_pol = PolarisationFrame('stokesIQUV')
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        else:
            self.vis_pol = PolarisationFrame('stokesI')
            self.image_pol = PolarisationFrame('stokesI')
            f = numpy.array([100.0])
        
        if dospectral:
            flux = numpy.array([f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency])
        else:
            flux = numpy.array([f])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        
        frequency_bag = bag.from_sequence([(i, self.frequency[i], self.channelwidth[i])
                                           for i, _ in enumerate(self.frequency)])
        
        def ingest_bag(f_bag, **kwargs):
            return ingest_unittest_visibility(frequency=[f_bag[1]], channel_bandwidth=[f_bag[2]], **kwargs)
        
        vis_bag = frequency_bag.map(ingest_bag, config=self.low, times=self.times,
                                    vis_pol=self.vis_pol, phasecentre=self.phasecentre, block=block)
        vis_bag = reify(vis_bag)
        
        model_bag = vis_bag.map(create_image_from_visibility,
                                npixel=self.params["npixel"],
                                cellsize=cellsize,
                                nchan=1,
                                polarisation_frame=self.image_pol)

        model_bag = reify(model_bag)

        def zero_image(im):
            im.data[...] = 0.0
            return im

        empty_model_bag = model_bag.map(zero_image)
        empty_model_bag = reify(empty_model_bag)
        
        # Make the components and fill the visibility and the model image
        flux_bag = bag.from_sequence([flux[i, :][numpy.newaxis, :] for i, _ in enumerate(self.frequency)])
        components_bag = empty_model_bag.map(create_unittest_components, flux_bag)
        if block:
            vis_bag = vis_bag.map(predict_skycomponent_visibility, components_bag)
        else:
            vis_bag = vis_bag.map(predict_skycomponent_visibility, components_bag)
        
        model_bag = model_bag.map(insert_skycomponent, components_bag)
        model_bag = reify(model_bag)
        model = list(model_bag)[0]

        # Calculate the model convolved with a Gaussian.
        self.cmodel = smooth_image(model)
        export_image_to_fits(model, '%s/test_imaging_bags_model.fits' % self.dir)
        export_image_to_fits(self.cmodel, '%s/test_imaging_bags_cmodel.fits' % self.dir)
        
#        if add_errors:
#            vis_bag = vis_bag.map(insert_unittest_errors, phase_error=1.0, amplitude_error=0.0, seed=180555)

        empty_model_bag = reify(empty_model_bag)
        vis_bag = reify(vis_bag)

        # For the bag processing, we need to convert to records, which provide meta data for bags
        def to_record(bg, fwin, key):
            return {'freqwin': fwin, key: bg}
        
        freqwin_bag = bag.range(freqwin, npartitions=freqwin)
        self.vis_record_bag = vis_bag.map(to_record, freqwin_bag, key='vis')
        self.vis_record_bag = reify(self.vis_record_bag)
        self.empty_model_record_bag = empty_model_bag.map(to_record, freqwin_bag, key='image')
        self.empty_model_record_bag = reify(self.empty_model_record_bag)

    def test_continuum_imaging_pipeline(self):
        self.actualSetUp(block=True)
        self.vis_slices = 51
        self.context = 'wstack'
        continuum_imaging_bag = \
            continuum_imaging_pipeline_bag(self.vis_record_bag, model_bag=self.empty_model_record_bag,
                                           context=self.context,
                                           vis_slices=self.vis_slices,
                                           niter=1000, fractional_threshold=0.1,
                                           threshold=2.0, nmajor=0, gain=0.1)
        if self.compute:
            clean, residual, restored = continuum_imaging_bag.compute()
            restored = restored.compute()[0]
            export_image_to_fits(restored, '%s/test_pipelines_continuum_imaging_bag_restored.fits' % self.dir)
            qa = qa_image(restored)
            assert numpy.abs(qa.data['max'] - 116.835113361) < 5.0, str(qa)
            assert numpy.abs(qa.data['min']) < 5.0, str(qa)
    
    def test_ical_pipeline(self):
        self.actualSetUp(block=True, add_errors=True)
        self.vis_slices = self.ntimes
        self.context = '2d'
        ical_bag = \
            ical_pipeline_bag(self.vis_record_bag, model_bag=self.empty_model_record_bag,
                              context=self.context,
                              vis_slices=self.vis_slices,
                              niter=1000, fractional_threshold=0.1,
                              threshold=2.0, nmajor=5, gain=0.1, first_selfcal=1,
                              global_solution=False)
        if self.compute:
            clean, residual, restored = ical_bag.compute()
            restored = restored.compute()[0]
            export_image_to_fits(restored, '%s/test_pipelines_ical_bag_restored.fits' % self.dir)
            qa = qa_image(restored)
            assert numpy.abs(qa.data['max'] - 114.907364114) < 5.0, str(qa)
            assert numpy.abs(qa.data['min'] + 0.529827321512) < 5.0, str(qa)


if __name__ == '__main__':
    unittest.main()
