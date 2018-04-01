""" Unit tests for calibration solution


"""
import numpy

import unittest

from astropy.coordinates import SkyCoord
import astropy.units as u

from arl.data.data_models import Skycomponent
from arl.data.polarisation import PolarisationFrame

from arl.calibration.calibration_control import calibrate_function, create_calibration_controls, apply_gaintable
from arl.calibration.operations import create_gaintable_from_blockvisibility, gaintable_summary
from arl.util.testing_support import create_named_configuration, simulate_gaintable
from arl.visibility.operations import divide_visibility
from arl.visibility.base import copy_visibility, create_blockvisibility
from arl.imaging import predict_skycomponent_visibility

import logging

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

    def test_calibrate_function(self):
        self.actualSetup('stokesI', 'stokesI', f=[100.0])
        # Prepare the corrupted visibility data
        gt = create_gaintable_from_blockvisibility(self.vis)
        log.info("Created gain table: %s" % (gaintable_summary(gt)))
        gt = simulate_gaintable(gt, phase_error=10.0, amplitude_error=0.1, timeslice='auto')
        bgt = simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.01, timeslice=1e5)
        original = copy_visibility(self.vis)
        self.vis = apply_gaintable(self.vis, bgt, vis_slices=1)
        self.vis = apply_gaintable(self.vis, gt, vis_slices=None)
        # Now get the control dictionary and calibrate
        controls = create_calibration_controls()
        controls['T']['first_selfcal']=0
        controls['B']['first_selfcal']=0
        calibrated_vis, gaintables = calibrate_function(self.vis, original, calibration_context='TB', controls=controls)
        residual = numpy.max(gaintables['T'].residual)
        assert residual < 3e-2, "Max T residual = %s" % (residual)
        residual = numpy.max(gaintables['B'].residual)
        assert residual < 6e-5, "Max B residual = %s" % (residual)



if __name__ == '__main__':
    unittest.main()
