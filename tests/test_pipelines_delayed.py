"""Unit tests for pipelines expressed via dask.delayed


"""

import logging
import os
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.graphs.execute import arlexecute

from arl.calibration.calibration_control import create_calibration_controls
from arl.data.polarisation import PolarisationFrame
from arl.image.operations import export_image_to_fits, smooth_image, qa_image
from arl.imaging import predict_skycomponent_visibility
from arl.pipelines.delayed import create_ical_pipeline_graph, create_continuum_imaging_pipeline_graph
from arl.skycomponent.operations import insert_skycomponent
from arl.util.testing_support import create_named_configuration, ingest_unittest_visibility, create_unittest_model, \
    create_unittest_components, insert_unittest_errors

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestPipelineGraphs(unittest.TestCase):
    
    def setUp(self):
        
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
    
    def actualSetUp(self, add_errors=False, freqwin=7, block=False, dospectral=True, dopol=False,
                    amp_errors=None, phase_errors=None, zerow=True):
        
        if amp_errors is None:
            amp_errors = {'T': 0.0, 'G': 0.01, 'B': 0.01}
        if phase_errors is None:
            phase_errors = {'T': 1.0, 'G': 0.1, 'B': 0.01}
        
        self.npixel = 256
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = freqwin
        self.vis_graph_list = list()
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
        self.vis_graph_list = [arlexecute.execute(ingest_unittest_visibility)(self.low,
                                                                   [self.frequency[i]],
                                                                   [self.channelwidth[i]],
                                                                   self.times,
                                                                   self.vis_pol,
                                                                   self.phasecentre, block=block,
                                                                   zerow=zerow)
                               for i, _ in enumerate(self.frequency)]
        
        self.model_graph = [arlexecute.execute(create_unittest_model, nout=freqwin)(self.vis_graph_list[0], self.image_pol,
                                                                         npixel=self.npixel)
                            for i, _ in enumerate(self.frequency)]
        
        self.components_graph = [arlexecute.execute(create_unittest_components)(self.model_graph[i], flux[i, :][numpy.newaxis, :])
                                 for i, _ in enumerate(self.frequency)]
        
        # Apply the LOW primary beam and insert into model
        self.model_graph = [arlexecute.execute(insert_skycomponent, nout=1)(self.model_graph[freqwin],
                                                                 self.components_graph[freqwin])
                            for freqwin, _ in enumerate(self.frequency)]
        
        self.vis_graph_list = [arlexecute.execute(predict_skycomponent_visibility)(self.vis_graph_list[freqwin],
                                                                        self.components_graph[freqwin])
                               for freqwin, _ in enumerate(self.frequency)]
        
        # Calculate the model convolved with a Gaussian.
        model = arlexecute.get(self.model_graph[0])
        self.cmodel = smooth_image(model)
        export_image_to_fits(model, '%s/test_imaging_delayed_model.fits' % self.dir)
        export_image_to_fits(self.cmodel, '%s/test_imaging_delayed_cmodel.fits' % self.dir)
        
        if add_errors and block:
            self.vis_graph_list = [arlexecute.execute(insert_unittest_errors)(self.vis_graph_list[i], amp_errors=amp_errors,
                                                                   phase_errors=phase_errors)
                                   for i, _ in enumerate(self.frequency)]
    
    def test_time_setup(self):
        self.actualSetUp()
    
    def test_continuum_imaging_pipeline(self):
        self.actualSetUp(add_errors=False, block=True)
        continuum_imaging_graph = \
            create_continuum_imaging_pipeline_graph(self.vis_graph_list, model_graph=self.model_graph, context='2d',
                                                    algorithm='mmclean',
                                                    facets=1,
                                                    niter=1000, fractional_threshold=0.1,
                                                    nmoments=3, nchan=self.freqwin,
                                                    threshold=2.0, nmajor=5, gain=0.1,
                                                    deconvolve_facets=4, deconvolve_overlap=16, deconvolve_taper='tukey')
        clean, residual, restored = arlexecute.get(continuum_imaging_graph)
        export_image_to_fits(clean[0], '%s/test_pipelines_continuum_imaging_pipeline_clean.fits' % self.dir)
        export_image_to_fits(residual[0][0],
                             '%s/test_pipelines_continuum_imaging_pipeline_residual.fits' % self.dir)
        export_image_to_fits(restored[0],
                             '%s/test_pipelines_continuum_imaging_pipeline_restored.fits' % self.dir)
        
        qa = qa_image(restored[0])
        assert numpy.abs(qa.data['max'] - 116.86978265) < 5.0, str(qa)
        assert numpy.abs(qa.data['min'] + 0.323425377573) < 5.0, str(qa)
    
    def test_ical_pipeline(self):
        amp_errors = {'T': 0.0, 'G': 0.00, 'B': 0.0}
        phase_errors = {'T': 1.0, 'G': 0.0, 'B': 0.0}
        self.actualSetUp(add_errors=True, block=True, amp_errors=amp_errors, phase_errors=phase_errors)
        
        controls = create_calibration_controls()
        
        controls['T']['first_selfcal'] = 1
        controls['G']['first_selfcal'] = 3
        controls['B']['first_selfcal'] = 4
        
        controls['T']['timescale'] = 'auto'
        controls['G']['timescale'] = 'auto'
        controls['B']['timescale'] = 1e5
        
        ical_graph = \
            create_ical_pipeline_graph(self.vis_graph_list, model_graph=self.model_graph, context='2d',
                                       calibration_context='T', controls=controls, do_selfcal=True,
                                       global_solution=False,
                                       algorithm='mmclean',
                                       facets=1,
                                       niter=1000, fractional_threshold=0.1,
                                       nmoments=3, nchan=self.freqwin,
                                       threshold=2.0, nmajor=5, gain=0.1,
                                       deconvolve_facets=4, deconvolve_overlap=16, deconvolve_taper='tukey')
        clean, residual, restored = arlexecute.get(ical_graph)
        export_image_to_fits(clean[0], '%s/test_pipelines_ical_pipeline_clean.fits' % self.dir)
        export_image_to_fits(residual[0][0], '%s/test_pipelines_ical_pipeline_residual.fits' % self.dir)
        export_image_to_fits(restored[0], '%s/test_pipelines_ical_pipeline_restored.fits' % self.dir)
        
        qa = qa_image(restored[0])
        assert numpy.abs(qa.data['max'] - 116.86978265) < 5.0, str(qa)
        assert numpy.abs(qa.data['min'] + 0.323425377573) < 5.0, str(qa)

if __name__ == '__main__':
    unittest.main()
