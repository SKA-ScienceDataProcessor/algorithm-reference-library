""" Unit tests for pipelines expressed via dask.bag


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

from arl.graphs.bags import invert_bag, predict_bag, deconvolve_bag, restore_bag, \
    residual_image_bag, predict_record_subtract, reify, selfcal_bag

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImagingBags(unittest.TestCase):
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
        flux_bag = bag.from_sequence([flux[i, :][numpy.newaxis, :]
                                      for i, _ in enumerate(self.frequency)])
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
        
        if add_errors:
            vis_bag = vis_bag.map(insert_unittest_errors, phase_error=1.0, amplitude_error=0.0, seed=180555)

        empty_model_bag = reify(empty_model_bag)
        vis_bag = reify(vis_bag)
        
        # For the bag processing, we need to convert to records, which provide meta data for bags
        def to_record(vis, f, key):
            return {'freqwin': f, key: vis}
        
        freqwin_bag = bag.range(freqwin, npartitions=freqwin)
        
        self.vis_record_bag = vis_bag.map(to_record, freqwin_bag, key='vis')
        self.vis_record_bag = reify(self.vis_record_bag)
        self.model_record_bag = model_bag.map(to_record, freqwin_bag, key='image')
        self.model_record_bag = reify(self.model_record_bag)
        self.empty_model_record_bag = empty_model_bag.map(to_record, freqwin_bag, key='image')
        self.empty_model_record_bag = reify(self.empty_model_record_bag)
    
    def test_invert_bag(self):
        self.actualSetUp()
        peaks = {'2d': 115.100462556, 'timeslice': 115.100462556, 'wstack': 115.100462556}
        vis_slices = {'2d': None, 'timeslice': 'auto', 'wstack': 101}
        for context in ['2d', 'timeslice', 'wstack']:
            dirty_bag = invert_bag(self.vis_record_bag, self.empty_model_record_bag, dopsf=False,
                                   context=context, normalize=True,
                                   vis_slices=vis_slices[context])
            dirty, sumwt = dirty_bag.compute()[0]['image']
            export_image_to_fits(dirty, '%s/test_bags_%s_dirty.fits' % (self.dir, context))
            qa = qa_image(dirty, context=context)
            
            assert numpy.abs(qa.data['max'] - peaks[context]) < 1.0e-2, str(qa)
    
    def test_predict_bag(self):
        self.actualSetUp()
        errors = {'2d': 28.0, 'timeslice': 8.89106002548, 'wstack': 8.89106002548}
        vis_slices = {'2d': None, 'timeslice': 'auto', 'wstack': 101}
        for context in ['wstack', 'timeslice']:
            model_vis_bag = predict_bag(self.vis_record_bag, self.model_record_bag, context,
                                        vis_slices=vis_slices[context])
            
            model_vis_bag = reify(model_vis_bag)
            error_vis_bag = self.vis_record_bag.map(predict_record_subtract, model_vis_bag)
            error_vis_bag = reify(error_vis_bag)
            error_image_bag = invert_bag(error_vis_bag, self.model_record_bag, dopsf=False,
                                         context=context, normalize=True, vis_slices=vis_slices[context])
            result = error_image_bag.compute()
            error_image = result[0]['image'][0]
            export_image_to_fits(error_image,
                                 '%s/test_bags_%s_predict_error_image.fits' % (self.dir, context))
            qa = qa_image(error_image, context='error image for %s' % context)
            assert numpy.abs(qa.data['max'] - errors[context]) < 0.1, str(qa)
    
    def test_deconvolve_bag(self):
        self.actualSetUp()
        context = 'wstack'
        vis_slices = {'2d': None, 'timeslice': 'auto', 'wstack': 101}
        dirty_bag = invert_bag(self.vis_record_bag, self.model_record_bag, dopsf=False, context=context,
                               normalize=True,
                               vis_slices=vis_slices[context])
        psf_bag = invert_bag(self.vis_record_bag, self.model_record_bag, dopsf=True, context=context,
                             normalize=True,
                             vis_slices=vis_slices[context])
        dirty_bag = reify(dirty_bag)
        psf_bag = reify(psf_bag)
        model_bag = deconvolve_bag(dirty_bag, psf_bag, self.empty_model_record_bag, niter=1000, gain=0.7,
                                   algorithm='msclean', threshold=0.01, window_shape=None)
        model = model_bag.compute()[0]['image']
        qa = qa_image(model, context=context)
        
        export_image_to_fits(model, '%s/test_bags_%s_deconvolve.fits' % (self.dir, context))
        
        assert numpy.abs(qa.data['max'] - 78.7691620819) < 0.1, str(qa)
        assert numpy.abs(qa.data['min'] + 2.68972448968) < 0.1, str(qa)
    
    def test_restore_bag(self):
        self.actualSetUp()
        
        peaks = {'wstack': 116.625014524}
        vis_slices = {'wstack': 101}
        context = 'wstack'
        dirty_bag = invert_bag(self.vis_record_bag, self.model_record_bag, dopsf=False, context=context,
                               normalize=True,
                               vis_slices=vis_slices[context])
        psf_bag = invert_bag(self.vis_record_bag, self.model_record_bag, dopsf=True, context=context,
                             normalize=True,
                             vis_slices=vis_slices[context])
        dirty_bag = reify(dirty_bag)
        psf_bag = reify(psf_bag)
        model_bag = deconvolve_bag(dirty_bag, psf_bag, self.empty_model_record_bag, niter=1000, gain=0.7,
                                   algorithm='msclean', threshold=0.01, window_shape=None)
        model_bag = reify(model_bag)
        
        res_image_bag = residual_image_bag(self.vis_record_bag, model_bag, context=context,
                                           vis_slices=vis_slices[context])
        res_image_bag = reify(res_image_bag)
        residual = res_image_bag.compute()[0]['image'][0]
        export_image_to_fits(residual, '%s/test_bags_%s_residual.fits' %
                             (self.dir, context))
        
        final_bag = restore_bag(model_bag, psf_bag, res_image_bag)
        final = final_bag.compute()[0]
        qa = qa_image(final, context=context)
        export_image_to_fits(final, '%s/test_bags_%s_restored.fits' %
                             (self.dir, context))
        assert numpy.abs(qa.data['max'] - peaks[context]) < 0.1, str(qa)
    
    def test_residual_image_bag(self):
        self.actualSetUp()
        context = 'wstack'
        vis_slices = {'wstack': 101}
        dirty_bag = invert_bag(self.vis_record_bag, self.empty_model_record_bag, dopsf=False, context=context,
                               normalize=True, vis_slices=vis_slices[context])
        psf_bag = invert_bag(self.vis_record_bag, self.empty_model_record_bag, dopsf=True, context=context,
                             normalize=True, vis_slices=vis_slices[context])
        dirty_bag = reify(dirty_bag)
        psf_bag = reify(psf_bag)
        model_bag = deconvolve_bag(dirty_bag, psf_bag, self.empty_model_record_bag, niter=1000,
                                   gain=0.1, algorithm='msclean',
                                   threshold=0.01, window_shape=None)
        residual_bag = residual_image_bag(self.vis_record_bag, model_bag, context=context,
                                          vis_slices=vis_slices[context])
        final = residual_bag.compute()[0]['image'][0]
        export_image_to_fits(final, '%s/test_bags_%s_residual.fits' % (self.dir, context))
        
        qa = qa_image(final, context=context)
        assert numpy.abs(qa.data['max'] - 5.7369326339) < 0.1, str(qa)

    def test_residual_image_bag_model(self):
        self.actualSetUp()
        context = 'wstack'
        vis_slices = {'wstack': 101}
        residual_bag = residual_image_bag(self.vis_record_bag, self.model_record_bag, context=context,
                                          vis_slices=vis_slices[context])
        final = residual_bag.compute()[0]['image'][0]
        export_image_to_fits(final, '%s/test_bags_%s_residual_image_bag.fits' % (self.dir, context))
        
        qa = qa_image(final, context=context)
        assert numpy.abs(qa.data['max'] - 8.89106002548) < 0.1, str(qa)
    
    @unittest.skip("Global solution not implemented yet")
    def test_selfcal_global_bag(self):
        self.actualSetUp(block=True)
        selfcal_vis_bag = selfcal_bag(self.vis_record_bag, self.model_record_bag, global_solution=True,
                                      context='timeslice', vis_slices=self.ntimes)
        dirty_bag = invert_bag(selfcal_vis_bag, self.model_record_bag,
                               dopsf=False, normalize=True, context='timeslice',
                               vis_slices=self.ntimes)
        if self.compute:
            dirty, sumwt = dirty_bag.compute()[0]['image']
            export_image_to_fits(dirty, '%s/test_imaging_bags_global_selfcal_dirty.fits'
                                 % (self.dir))
            qa = qa_image(dirty)
            
            assert numpy.abs(qa.data['max'] - 112.282380843) < 0.1, str(qa)
            assert numpy.abs(qa.data['min'] + 2.38763650521) < 0.1, str(qa)
    
    def test_selfcal_nonglobal_bag(self):
        
        self.actualSetUp(block=True)
        selfcal_vis_bag = selfcal_bag(self.vis_record_bag, self.model_record_bag, global_solution=False,
                                      context='timeslice', vis_slices=self.ntimes)
        
        dirty_bag = invert_bag(selfcal_vis_bag, self.model_record_bag,
                               dopsf=False, normalize=True, context='timeslice', vis_slices=self.ntimes)
        if self.compute:
            dirty, sumwt = dirty_bag.compute()[0]['image']
            export_image_to_fits(dirty, '%s/test_imaging_bags_nonglobal_selfcal_dirty.fits' % (self.dir))
            qa = qa_image(dirty)
            
            assert numpy.abs(qa.data['max'] - 112.282380843) < 0.1, str(qa)
            assert numpy.abs(qa.data['min'] + 2.38763650521) < 0.1, str(qa)
