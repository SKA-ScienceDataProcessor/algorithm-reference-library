"""Unit tests for pipelines expressed via dask.delayed


"""

import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame

from workflows.serial.pipelines.pipeline_serial import ical_list_serial_workflow, continuum_imaging_list_serial_workflow
from wrappers.serial.calibration.calibration_control import create_calibration_controls
from wrappers.serial.image.operations import export_image_to_fits, qa_image, smooth_image
from wrappers.serial.imaging.base import predict_skycomponent_visibility
from wrappers.serial.simulation.testing_support import create_named_configuration, ingest_unittest_visibility, \
    create_unittest_model, create_unittest_components, insert_unittest_errors
from wrappers.serial.skycomponent.operations import insert_skycomponent
from wrappers.serial.visibility.coalesce import convert_blockvisibility_to_visibility
from wrappers.serial.simulation.testing_support import simulate_gaintable
from wrappers.serial.calibration.operations import create_gaintable_from_blockvisibility, apply_gaintable

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestPipelines(unittest.TestCase):
    
    def setUp(self):
        
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        self.persist = True
    
    def tearDown(self):
        pass
    
    def actualSetUp(self, add_errors=False, nfreqwin=7, dospectral=True, dopol=False, zerow=True):
        
        self.npixel = 512
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = nfreqwin
        self.vis_list = list()
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        
        if self.freqwin > 1:
            self.channelwidth = numpy.array(self.freqwin * [self.frequency[1] - self.frequency[0]])
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
        self.blockvis_list = [ingest_unittest_visibility(self.low,
                                                         [self.frequency[i]],
                                                         [self.channelwidth[i]],
                                                         self.times,
                                                         self.vis_pol,
                                                         self.phasecentre, block=True,
                                                         zerow=zerow)
                              for i in range(nfreqwin)]
        
        self.vis_list = [convert_blockvisibility_to_visibility(bv) for bv in self.blockvis_list]
        
        self.model_imagelist = [
            create_unittest_model(self.vis_list[i], self.image_pol, npixel=self.npixel, cellsize=0.0005)
            for i in range(nfreqwin)]
        
        self.components_list = [create_unittest_components(self.model_imagelist[freqwin],
                                                           flux[freqwin, :][numpy.newaxis, :])
                                for freqwin, m in enumerate(self.model_imagelist)]
        
        self.blockvis_list = [
            predict_skycomponent_visibility(self.blockvis_list[freqwin], self.components_list[freqwin])
            for freqwin, _ in enumerate(self.blockvis_list)]
        
        self.model_imagelist = [insert_skycomponent(self.model_imagelist[freqwin], self.components_list[freqwin])
                                for freqwin in range(nfreqwin)]
        model = self.model_imagelist[0]
        self.cmodel = smooth_image(model)
        if self.persist:
            export_image_to_fits(model, '%s/test_imaging_serial_model.fits' % self.dir)
            export_image_to_fits(self.cmodel, '%s/test_imaging_serial_cmodel.fits' % self.dir)
        
        if add_errors:
            gt = create_gaintable_from_blockvisibility(self.blockvis_list[0])
            gt = simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.0, smooth_channels=1,
                       leakage=0.0, seed=180555)
            self.blockvis_list = [apply_gaintable(self.blockvis_list[i], gt)
                                  for i in range(self.freqwin)]
        
        self.vis_list = [convert_blockvisibility_to_visibility(bv) for bv in self.blockvis_list]
        
        self.model_imagelist = [
            create_unittest_model(self.vis_list[i], self.image_pol, npixel=self.npixel, cellsize=0.0005)
            for i in range(nfreqwin)]
    
    def test_time_setup(self):
        self.actualSetUp(add_errors=True)
    
    def test_continuum_imaging_pipeline(self):
        self.actualSetUp(add_errors=False, zerow=True)
        clean, residual, restored = \
            continuum_imaging_list_serial_workflow(self.vis_list,
                                                   model_imagelist=self.model_imagelist,
                                                   context='2d',
                                                   algorithm='mmclean', facets=1,
                                                   scales=[0, 3, 10],
                                                   niter=1000, fractional_threshold=0.1, threshold=0.1,
                                                   nmoment=3,
                                                   nmajor=5, gain=0.1,
                                                   deconvolve_facets=4, deconvolve_overlap=32,
                                                   deconvolve_taper='tukey', psf_support=64)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre], '%s/test_pipelines_continuum_imaging_pipeline_serial_clean.fits' % self.dir)
            export_image_to_fits(residual[centre][0],
                                '%s/test_pipelines_continuum_imaging_pipeline_serial_residual.fits' % self.dir)
            export_image_to_fits(restored[centre],
                                '%s/test_pipelines_continuum_imaging_pipeline_serial_restored.fits' % self.dir)
        
        qa = qa_image(restored[centre])
        assert numpy.abs(qa.data['max'] - 100.13762476849081) < 1.0, str(qa)
        assert numpy.abs(qa.data['min'] + 0.03627273884170454) < 1.0, str(qa)
    
    def test_ical_pipeline(self):
        self.actualSetUp(add_errors=True)
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 1
        controls['T']['timescale'] = 'auto'
    
        clean, residual, restored = \
            ical_list_serial_workflow(self.vis_list,
                                      model_imagelist=self.model_imagelist,
                                      context='2d',
                                      algorithm='mmclean', facets=1,
                                      scales=[0, 3, 10],
                                      niter=1000, fractional_threshold=0.1, threshold=0.1,
                                      nmoment=3,
                                      nmajor=5, gain=0.1,
                                      deconvolve_facets=4, deconvolve_overlap=32,
                                      deconvolve_taper='tukey', psf_support=64,
                                      calibration_context='T', controls=controls, do_selfcal=True,
                                      global_solution=False)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre], '%s/test_pipelines_ical_pipeline_serial_clean.fits' % self.dir)
            export_image_to_fits(residual[centre][0], '%s/test_pipelines_ical_pipeline_serial_residual.fits' % self.dir)
            export_image_to_fits(restored[centre], '%s/test_pipelines_ical_pipeline_serial_restored.fits' % self.dir)
        
        qa = qa_image(restored[centre])
        assert numpy.abs(qa.data['max'] - 99.32729396999524) < 1.0, str(qa)
        assert numpy.abs(qa.data['min'] + 0.6501547522800477) < 1.0, str(qa)
    
    def test_ical_pipeline_global(self):
        self.actualSetUp(add_errors=True)
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 1
        controls['T']['timescale'] = 'auto'
        
        clean, residual, restored = \
            ical_list_serial_workflow(self.vis_list,
                                      model_imagelist=self.model_imagelist,
                                      context='2d',
                                      algorithm='mmclean', facets=1,
                                      scales=[0, 3, 10],
                                      niter=1000, fractional_threshold=0.1, threshold=0.1,
                                      nmoment=3,
                                      nmajor=5, gain=0.1,
                                      deconvolve_facets=4, deconvolve_overlap=32,
                                      deconvolve_taper='tukey', psf_support=64,
                                      calibration_context='T', controls=controls, do_selfcal=True,
                                      global_solution=True)
        
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre], '%s/test_pipelines_ical_global_pipeline_serial_clean.fits' % self.dir)
            export_image_to_fits(residual[centre][0], '%s/test_pipelines_ical_global_pipeline_serial_residual.fits' % self.dir)
            export_image_to_fits(restored[centre], '%s/test_pipelines_ical_global_pipeline_serial_restored.fits' % self.dir)
        
        qa = qa_image(restored[centre])
        assert numpy.abs(qa.data['max'] - 98.92656340122159) < 1.0, str(qa)
        assert numpy.abs(qa.data['min'] + 0.7024492707920869) < 1.0, str(qa)


if __name__ == '__main__':
    unittest.main()
