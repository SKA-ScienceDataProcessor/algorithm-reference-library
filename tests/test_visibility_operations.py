"""Unit tests for visibility operations

realtimcornwell@gmail.com
"""

import unittest

from numpy.testing import assert_allclose

from arl.fourier_transforms.ftprocessor import *
from arl.util.testing_support import create_named_configuration
from arl.util.run_unittests import run_unittests

from arl.visibility.operations import *


class TestVisibilityOperations(unittest.TestCase):
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
        #        self.compabsdirection = SkyCoord(ra=+182 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox=2000.0)
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)
        self.comp = Skycomponent(direction=self.compreldirection, frequency=self.frequency, flux=self.flux)
    
    def test_create_visibility(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                     weight=1.0, npol=1)
        assert self.vis.nvis == len(self.vis.time)
        assert self.vis.nvis == len(self.vis.frequency)

    def test_append_visibility(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                     weight=1.0, npol=1)
        othertimes = (numpy.pi / 43200.0) * numpy.arange(300.0, 600.0, 30.0)
        self.othervis = create_visibility(self.lowcore, othertimes, self.frequency, phasecentre=self.phasecentre,
                                     weight=1.0, npol=1)
        self.vis = append_visibility(self.vis, self.othervis)
        assert self.vis.nvis == len(self.vis.time)
        assert self.vis.nvis == len(self.vis.frequency)

    def test_copy_visibility(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     phasecentre=self.phasecentre, weight=1.0, npol=self.flux.shape[1])
        vis = copy_visibility(self.vis)
        self.vis.data['vis'] = 0.0
        vis.data['vis'] = 1.0
        log.debug(self.vis.data['vis'][0].real)
        assert (vis.data['vis'][0].real == 1.0)
        assert (self.vis.data['vis'][0].real == 0.0)
    
    def test_visibilitysum(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                     weight=1.0, npol=self.flux.shape[1])
        self.vismodel = predict_skycomponent_visibility(self.vis, self.comp)
        # Sum the visibilities in the correct_visibility direction. This is limited by numerical precision
        summedflux, weight = sum_visibility(self.vismodel, self.compreldirection)
        assert_allclose(self.flux, summedflux, rtol=1e-7)
    
    def test_phase_rotation_identity(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                     weight=1.0, npol=self.flux.shape[1])
        self.vismodel = predict_skycomponent_visibility(self.vis, self.comp)
        newphasecenters = [SkyCoord(182, -35, unit=u.deg), SkyCoord(182, -30, unit=u.deg),
                           SkyCoord(177, -30, unit=u.deg), SkyCoord(176, -35, unit=u.deg),
                           SkyCoord(216, -35, unit=u.deg), SkyCoord(180, -70, unit=u.deg)]
        for newphasecentre in newphasecenters:
            # Phase rotating back should not make a difference
            original_vis = self.vismodel.vis
            original_uvw = self.vismodel.uvw
            rotatedvis = phaserotate_visibility(phaserotate_visibility(self.vismodel, newphasecentre, tangent=False),
                                                self.phasecentre, tangent=False)
            assert_allclose(rotatedvis.uvw, original_uvw, rtol=1e-7)
            assert_allclose(rotatedvis.vis, original_vis, rtol=1e-7)
    
    def test_phase_rotation(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                     weight=1.0, npol=self.flux.shape[1])
        self.vismodel = predict_skycomponent_visibility(self.vis, self.comp)
        # Predict visibilities with new phase centre independently
        ha_diff = -(self.compabsdirection.ra - self.phasecentre.ra).to(u.rad).value
        vispred = create_visibility(self.lowcore, self.times + ha_diff, self.frequency,
                                    phasecentre=self.compabsdirection, weight=1.0, npol=self.flux.shape[1])
        vismodel2 = predict_skycomponent_visibility(vispred, self.comp)
        
        # Should yield the same results as rotation
        rotatedvis = phaserotate_visibility(self.vismodel, newphasecentre=self.compabsdirection, tangent=False)
        assert_allclose(rotatedvis.vis, vismodel2.vis, rtol=1e-7)
        assert_allclose(rotatedvis.uvw, vismodel2.uvw, rtol=1e-7)


if __name__ == '__main__':
    run_unittests()
