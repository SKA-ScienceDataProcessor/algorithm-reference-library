"""Unit tests for visibility calibration


"""

import numpy
import logging
import unittest

from astropy.coordinates import SkyCoord
import astropy.units as u
from arl.calibration.operations import gaintable_summary, apply_gaintable, create_gaintable_from_blockvisibility

from arl.data.data_models import Skycomponent
from arl.data.polarisation import PolarisationFrame
from arl.util.testing_support import create_named_configuration, simulate_gaintable
from arl.visibility.base import copy_visibility, create_blockvisibility
from arl.imaging import predict_skycomponent_blockvisibility

log = logging.getLogger(__name__)

class TestCalibrationOperations(unittest.TestCase):
    
    def setUp(self):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        vnchan = 3
        self.frequency = numpy.linspace(1.0e8, 1.1e8, vnchan)
        self.channel_bandwidth = numpy.array(vnchan * [self.frequency[1] - self.frequency[0]])
        
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])

        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        
    def actualSetup(self, sky_pol_frame='stokesIQUV', data_pol_frame='linear'):
        self.comp = Skycomponent(direction=self.compabsdirection, frequency=self.frequency, flux=self.flux,
                                 polarisation_frame=PolarisationFrame(sky_pol_frame))
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                     channel_bandwidth=self.channel_bandwidth,
                                          weight=1.0, polarisation_frame=PolarisationFrame(data_pol_frame))
        self.vis = predict_skycomponent_blockvisibility(self.vis, self.comp)

    def test_create_gaintable_from_visibility(self):
        for spf, dpf in[('stokesIQUV', 'linear'), ('stokesIQUV', 'circular') ]:
            self.actualSetup(spf, dpf)
            gt = create_gaintable_from_blockvisibility(self.vis)
            log.info("Created gain table: %s" % (gaintable_summary(gt)))
            gt = simulate_gaintable(gt, phase_error=0.1)
            original = copy_visibility(self.vis)
            vis = apply_gaintable(self.vis, gt)
            assert numpy.max(numpy.abs(vis.vis-original.vis)) > 0.0

    def test_apply_gaintable_only(self):
        for spf, dpf in[('stokesIQUV', 'linear'), ('stokesIQUV', 'circular') ]:
            self.actualSetup(spf, dpf)
            gt = create_gaintable_from_blockvisibility(self.vis)
            log.info("Created gain table: %s" % (gaintable_summary(gt)))
            gt = simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.01)
            original = copy_visibility(self.vis)
            vis = apply_gaintable(self.vis, gt)
            error = numpy.max(numpy.abs(vis.vis-original.vis))
            assert error > 10.0, "Error = %f" % (error)

    def test_apply_gaintable_and_inverse_phase_only(self):
        for spf, dpf in[('stokesIQUV', 'linear'), ('stokesIQUV', 'circular') ]:
            self.actualSetup(spf, dpf)
            gt = create_gaintable_from_blockvisibility(self.vis)
            log.info("Created gain table: %s" % (gaintable_summary(gt)))
            gt = simulate_gaintable(gt, phase_error=0.1)
            original = copy_visibility(self.vis)
            vis = apply_gaintable(self.vis, gt)
            vis = apply_gaintable(self.vis, gt, inverse=True)
            error = numpy.max(numpy.abs(vis.vis-original.vis))
            assert error < 1e-12, "Error = %s" % (error)


    def test_apply_gaintable_and_inverse_both(self):
        for spf, dpf in[('stokesIQUV', 'linear'), ('stokesIQUV', 'circular') ]:
            self.actualSetup(spf, dpf)
            gt = create_gaintable_from_blockvisibility(self.vis)
            log.info("Created gain table: %s" % (gaintable_summary(gt)))
            gt = simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.1)
            original = copy_visibility(self.vis)
            vis = apply_gaintable(self.vis, gt)
            vis = apply_gaintable(self.vis, gt, inverse=True)
            error = numpy.max(numpy.abs(vis.vis-original.vis))
            assert error < 1e-12, "Error = %s" % (error)
    

if __name__ == '__main__':
    unittest.main()
