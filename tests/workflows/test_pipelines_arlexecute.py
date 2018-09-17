"""Unit tests for pipelines expressed via dask.delayed


"""

import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame

from tests.workflows import ARLExecuteTestCase
from wrappers.arlexecute.calibration.calibration_control import create_calibration_controls
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from workflows.arlexecute.pipelines.pipeline_arlexecute import ical_list_arlexecute_workflow, continuum_imaging_list_arlexecute_workflow
from wrappers.arlexecute.image.operations import export_image_to_fits, qa_image, smooth_image
from wrappers.arlexecute.imaging.base import predict_skycomponent_visibility
from wrappers.arlexecute.skycomponent.operations import insert_skycomponent
from wrappers.arlexecute.simulation.testing_support import create_named_configuration, ingest_unittest_visibility, \
    create_unittest_model, \
    create_unittest_components, insert_unittest_errors

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestPipelineGraphs(ARLExecuteTestCase, unittest.TestCase):
    
    def setUp(self):
        super(TestPipelineGraphs, self).setUp()
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')

    def actualSetUp(self, add_errors=False, freqwin=5, block=False, dospectral=True, dopol=False,
                    amp_errors=None, phase_errors=None, zerow=True):
        
        if amp_errors is None:
            amp_errors = {'T': 0.0, 'G': 0.01, 'B': 0.01}
        if phase_errors is None:
            phase_errors = {'T': 1.0, 'G': 0.1, 'B': 0.01}
        
        self.npixel = 512
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = freqwin
        self.vis_list = list()
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
        self.vis_list = [arlexecute.execute(ingest_unittest_visibility)(self.low,
                                                                        [self.frequency[i]],
                                                                        [self.channelwidth[i]],
                                                                        self.times,
                                                                        self.vis_pol,
                                                                        self.phasecentre, block=block,
                                                                        zerow=zerow)
                         for i, _ in enumerate(self.frequency)]
        
        self.model_imagelist = [
            arlexecute.execute(create_unittest_model, nout=freqwin)(self.vis_list[0], self.image_pol,
                                                                    npixel=self.npixel)
            for i, _ in enumerate(self.frequency)]
        
        self.components_list = [
            arlexecute.execute(create_unittest_components)(self.model_imagelist[i], flux[i, :][numpy.newaxis, :])
            for i, _ in enumerate(self.frequency)]
        
        # Apply the LOW primary beam and insert into model
        self.model_imagelist = [arlexecute.execute(insert_skycomponent, nout=1)(self.model_imagelist[freqwin],
                                                                                self.components_list[freqwin])
                                for freqwin, _ in enumerate(self.frequency)]
        
        self.vis_list = [arlexecute.execute(predict_skycomponent_visibility)(self.vis_list[freqwin],
                                                                             self.components_list[freqwin])
                         for freqwin, _ in enumerate(self.frequency)]
        
        # Calculate the model convolved with a Gaussian.
        model = arlexecute.compute(self.model_imagelist[0], sync=True)
        self.cmodel = smooth_image(model)
        export_image_to_fits(model, '%s/test_imaging_delayed_model.fits' % self.dir)
        export_image_to_fits(self.cmodel, '%s/test_imaging_delayed_cmodel.fits' % self.dir)
        
        if add_errors and block:
            self.vis_list = [
                arlexecute.execute(insert_unittest_errors)(self.vis_list[i], amp_errors=amp_errors,
                                                           phase_errors=phase_errors)
                for i, _ in enumerate(self.frequency)]
            
    def test_time_setup(self):
        self.actualSetUp()
    
    def test_continuum_imaging_pipeline(self):
        self.actualSetUp(add_errors=False, block=True)
        continuum_imaging_list = \
            continuum_imaging_list_arlexecute_workflow(self.vis_list, model_imagelist=self.model_imagelist, context='2d',
                                                       algorithm='mmclean', facets=1,
                                                       scales=[0, 3, 10],
                                                       niter=1000, fractional_threshold=0.1,
                                                       nmoments=2, nchan=self.freqwin,
                                                       threshold=2.0, nmajor=5, gain=0.1,
                                                       deconvolve_facets=8, deconvolve_overlap=16,
                                                       deconvolve_taper='tukey')
        clean, residual, restored = arlexecute.compute(continuum_imaging_list, sync=True)
        export_image_to_fits(clean[0], '%s/test_pipelines_continuum_imaging_pipeline_clean.fits' % self.dir)
        export_image_to_fits(residual[0][0],
                             '%s/test_pipelines_continuum_imaging_pipeline_residual.fits' % self.dir)
        export_image_to_fits(restored[0],
                             '%s/test_pipelines_continuum_imaging_pipeline_restored.fits' % self.dir)
        
        qa = qa_image(restored[0])
        assert numpy.abs(qa.data['max'] - 116.9) < 1.0, str(qa)
        assert numpy.abs(qa.data['min'] + 0.118) < 1.0, str(qa)
    
    def test_ical_pipeline(self):
        amp_errors = {'T': 0.0, 'G': 0.00, 'B': 0.0}
        phase_errors = {'T': 0.1, 'G': 0.0, 'B': 0.0}
        self.actualSetUp(add_errors=True, block=True, amp_errors=amp_errors, phase_errors=phase_errors)
        
        controls = create_calibration_controls()
        
        controls['T']['first_selfcal'] = 1
        controls['G']['first_selfcal'] = 3
        controls['B']['first_selfcal'] = 4
        
        controls['T']['timescale'] = 'auto'
        controls['G']['timescale'] = 'auto'
        controls['B']['timescale'] = 1e5
        
        ical_list = \
            ical_list_arlexecute_workflow(self.vis_list, model_imagelist=self.model_imagelist, context='2d',
                                          calibration_context='T', controls=controls, do_selfcal=True,
                                          global_solution=False,
                                          algorithm='mmclean',
                                          facets=1,
                                          scales=[0, 3, 10],
                                          niter=1000, fractional_threshold=0.1,
                                          nmoments=2, nchan=self.freqwin,
                                          threshold=2.0, nmajor=5, gain=0.1,
                                          deconvolve_facets=8, deconvolve_overlap=16, deconvolve_taper='tukey')
        clean, residual, restored = arlexecute.compute(ical_list, sync=True)
        export_image_to_fits(clean[0], '%s/test_pipelines_ical_pipeline_clean.fits' % self.dir)
        export_image_to_fits(residual[0][0], '%s/test_pipelines_ical_pipeline_residual.fits' % self.dir)
        export_image_to_fits(restored[0], '%s/test_pipelines_ical_pipeline_restored.fits' % self.dir)
        
        qa = qa_image(restored[0])
        assert numpy.abs(qa.data['max'] - 116.9) < 1.0, str(qa)
        assert numpy.abs(qa.data['min'] + 0.118) < 1.0, str(qa)


if __name__ == '__main__':
    unittest.main()
