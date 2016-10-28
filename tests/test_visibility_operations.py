"""Unit tests for visibility operations

realtimcornwell@gmail.com
"""

import unittest

from arl.fourier_transforms.ftprocessor import *
from astropy import units as u
from numpy.testing import assert_allclose

from skymodel.skymodel_operations import create_skycomponent
from skymodel.skymodel_operations import create_skymodel_from_component
from util.testing_support import create_named_configuration
from visibility.visibility_operations import *


class TestVisibilityOperations(unittest.TestCase):
    def setUp(self):
        self.params = {'wstep': 10.0, 'npixel': 512, 'cellsize': 0.0002, 'spectral_mode': 'channel'}
        
        self.vlaa = create_named_configuration('VLAA')
        self.vlaa.data['xyz'] *= 1.0 / 30.0
        self.times = numpy.arange(-3.0, +3.0, 6.0 / 60.0) * numpy.pi / 12.0
        self.frequency = numpy.arange(1.0e8, 1.50e8, 2.0e7)
        
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=+35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.compabsdirection = SkyCoord(ra=+17.0 * u.deg, dec=+36.5 * u.deg, frame='icrs', equinox=2000.0)
        # TODO: convert entire mechanism to absolute coordinates
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)
        self.m31comp = create_skycomponent(flux=self.flux,
                                           frequency=self.frequency, direction=self.compreldirection)
        self.m31sm = create_skymodel_from_component(self.m31comp)
        
        vispred = create_visibility(self.vlaa, self.times, self.frequency,
                                    weight=1.0, phasecentre=self.phasecentre,
                                    params=self.params)
        self.vismodel = predict_skycomponent_visibility(vispred, self.m31comp, self.params)
    
    def test_visibilitysum(self):
        # Sum the visibilities in the correct_visibility direction. This is limited by numerical precision
        summedflux, weight = sum_visibility(self.vismodel, self.compreldirection)
        assert_allclose(self.flux, summedflux, rtol=1e-7)
    
    def test_phase_rotation_identity(self):
        for newphasecentre in [SkyCoord(17, 35, unit=u.deg), SkyCoord(17, 30, unit=u.deg),
                               SkyCoord(12, 30, unit=u.deg), SkyCoord(11, 35, unit=u.deg),
                               SkyCoord(51, 35, unit=u.deg), SkyCoord(15, 70, unit=u.deg)]:
            # Phase rotating back should not make a difference
            original_vis = self.vismodel.vis
            original_uvw = self.vismodel.uvw
            rotatedvis = phaserotate_visibility(
                phaserotate_visibility(self.vismodel, newphasecentre),
                self.phasecentre)
            assert_allclose(rotatedvis.uvw, original_uvw, rtol=1e-10)
            assert_allclose(rotatedvis.vis, original_vis, rtol=1e-10)
    
    def test_phase_rotation(self):
        # Predict visibilities with new phase centre independently
        ha_diff = -(self.compabsdirection.ra - self.phasecentre.ra).to(u.rad).value
        vispred = create_visibility(self.vlaa, self.times + ha_diff, self.frequency,
                                    weight=1.0, phasecentre=self.compabsdirection,
                                    params=self.params)
        vismodel2 = predict_skycomponent_visibility(vispred, self.m31comp, self.params)
        
        # Should yield the same results as rotation
        rotatedvis = phaserotate_visibility(self.vismodel, self.compabsdirection)
        assert_allclose(rotatedvis.uvw, vismodel2.uvw, rtol=1e-10)
        assert_allclose(rotatedvis.vis, vismodel2.vis, rtol=1e-10)


if __name__ == '__main__':
    import sys
    import logging
    
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    unittest.main()
