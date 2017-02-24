"""Unit tests for visibility calibrations

realtimcornwell@gmail.com
"""
import unittest

from numpy.testing import assert_allclose

from arl.fourier_transforms.ftprocessor import *
from arl.util.testing_support import create_named_configuration, simulate_gaintable
from arl.util.run_unittests import run_unittests
from arl.calibration.gaintable import *
from arl.visibility.operations import create_blockvisibility


class TestCalibration(unittest.TestCase):
    
    def setUp(self):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, 3)
        
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.comp = Skycomponent(direction=self.compabsdirection, frequency=self.frequency, flux=self.flux)
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                     weight=1.0, npol=1)
        self.vis.data['vis'][...] = 1+0j

    def test_create_gaintable_from_visibility(self):
        gt = create_gaintable_from_blockvisibility(self.vis)
        log.info("Created gain table: %s" % (gaintable_summary(gt)))
        gt = simulate_gaintable(gt, phase_error=0.1)
        original = copy_visibility(self.vis)
        vis = apply_gaintable(self.vis, gt)
        assert numpy.max(numpy.abs(vis.vis-original.vis)) > 0.0

if __name__ == '__main__':
    unittest.main()
