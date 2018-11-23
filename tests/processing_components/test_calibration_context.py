""" Unit tests for calibration solution


"""
import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.memory_data_models import Skycomponent
from data_models.polarisation import PolarisationFrame

from processing_components.calibration.calibration_control import calibrate_function, create_calibration_controls, apply_gaintable
from processing_components.calibration.operations import create_gaintable_from_blockvisibility, gaintable_summary
from processing_components.imaging.base import predict_skycomponent_visibility
from processing_components.simulation.testing_support import create_named_configuration, simulate_gaintable
from processing_components.visibility.base import copy_visibility, create_blockvisibility

log = logging.getLogger(__name__)


class TestCalibrationContext(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(180555)
    
    def actualSetup(self, sky_pol_frame='stokesIQUV', data_pol_frame='linear', f=None, vnchan=3):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.linspace(0.0, 30.0, 3)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, vnchan)
        self.channel_bandwidth = numpy.array(vnchan * [self.frequency[1] - self.frequency[0]])
        
        if f is None:
            f = [100.0, 50.0, -10.0, 40.0]
        
        if sky_pol_frame == 'stokesI':
            f = [100.0]
        
        self.flux = numpy.outer(numpy.array([numpy.power(freq / 1e8, -0.7) for freq in self.frequency]), f)
        
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.comp = Skycomponent(direction=self.compabsdirection, frequency=self.frequency, flux=self.flux,
                                 polarisation_frame=PolarisationFrame(sky_pol_frame))
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                          channel_bandwidth=self.channel_bandwidth, weight=1.0,
                                          polarisation_frame=PolarisationFrame(data_pol_frame))
        self.vis = predict_skycomponent_visibility(self.vis, self.comp)
    
    def test_calibrate_T_function(self):
        self.actualSetup('stokesI', 'stokesI', f=[100.0])
        # Prepare the corrupted visibility data_models
        gt = create_gaintable_from_blockvisibility(self.vis)
        log.info("Created gain table: %s" % (gaintable_summary(gt)))
        gt = simulate_gaintable(gt, phase_error=10.0, amplitude_error=0.0)
        original = copy_visibility(self.vis)
        self.vis = apply_gaintable(self.vis, gt, vis_slices=None)
        # Now get the control dictionary and calibrate
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        calibrated_vis, gaintables = calibrate_function(self.vis, original, calibration_context='T',
                                                        controls=controls)
        residual = numpy.max(gaintables['T'].residual)
        assert residual < 1e-8, "Max T residual = %s" % (residual)


    def test_calibrate_TG_function(self):
        self.actualSetup('stokesI', 'stokesI', f=[100.0])
        # Prepare the corrupted visibility data_models
        gt = create_gaintable_from_blockvisibility(self.vis)
        log.info("Created gain table: %s" % (gaintable_summary(gt)))
        gt = simulate_gaintable(gt, phase_error=10.0, amplitude_error=0.0)
        original = copy_visibility(self.vis)
        self.vis = apply_gaintable(self.vis, gt, vis_slices=None)
        # Now get the control dictionary and calibrate
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        controls['G']['first_selfcal'] = 0
        calibrated_vis, gaintables = calibrate_function(self.vis, original, calibration_context='TG',
                                                        controls=controls)
        residual = numpy.max(gaintables['T'].residual)
        residual = numpy.max(gaintables['G'].residual)
        assert residual < 1e-8, "Max T residual = %s" % residual


if __name__ == '__main__':
    unittest.main()
