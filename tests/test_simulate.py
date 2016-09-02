from crocodile.simulate import *

import unittest
import numpy as np
from numpy.testing import assert_allclose


class TestCoordinates(unittest.TestCase):
    def test_xyz_at_latitude(self):
        def transform(x, y, z, lat):
            """

            :param x:
            :param y:
            :param z:
            :param lat:
            :returns:
            """
            res = xyz_at_latitude(np.array([x, y, z]), np.radians(lat))
            assert_allclose(np.linalg.norm(res), np.linalg.norm([x, y, z]))
            return res

        # At the north pole the zenith is the celestial north
        assert_allclose(transform(1, 0, 0, 90), [1, 0, 0], atol=1e-15)
        assert_allclose(transform(0, 1, 0, 90), [0, 1, 0], atol=1e-15)
        assert_allclose(transform(0, 0, 1, 90), [0, 0, 1], atol=1e-15)

        # At the equator the zenith is negative Y
        assert_allclose(transform(1, 0, 0, 0), [1, 0, 0], atol=1e-15)
        assert_allclose(transform(0, 1, 0, 0), [0, 0, 1], atol=1e-15)
        assert_allclose(transform(0, 0, 1, 0), [0, -1, 0], atol=1e-15)

        # At the south pole we have flipped Y and Z
        assert_allclose(transform(1, 0, 0, -90), [1, 0, 0], atol=1e-15)
        assert_allclose(transform(0, 1, 0, -90), [0, -1, 0], atol=1e-15)
        assert_allclose(transform(0, 0, 1, -90), [0, 0, -1], atol=1e-15)

    def test_xyz_to_uvw(self):
        def transform(x, y, z, ha, dec):
            """

            :param x:
            :param y:
            :param z:
            :param ha:
            :param dec:
            :returns:
            """
            res = xyz_to_uvw(np.array([x, y, z]), np.radians(ha), np.radians(dec))
            assert_allclose(np.linalg.norm(res), np.linalg.norm([x, y, z]))
            return res

        # Derived from http://casa.nrao.edu/Memos/CoordConvention.pdf

        # 1. For ha=0,dec=90, we should have UVW=XYZ
        assert_allclose(transform(0, 0, 1, 0, 90), [0, 0, 1], atol=1e-15)
        assert_allclose(transform(0, 1, 0, 0, 90), [0, 1, 0], atol=1e-15)
        assert_allclose(transform(1, 0, 0, 0, 90), [1, 0, 0], atol=1e-15)

        # Extra test: For dec=90, we always have Z=W
        assert_allclose(transform(0, 0, 1, -90, 90), [0, 0, 1], atol=1e-15)
        assert_allclose(transform(0, 0, 1, 90, 90), [0, 0, 1], atol=1e-15)

        # 2. V and W are always on a Great circle with the NCP

        # ... we need the inverse transform, I guess?

        # 3. when W is on the local meridian (hour angle 0), U points
        #    east (positive X)
        assert_allclose(transform(1, 0, 0, 0, 0), [1, 0, 0], atol=1e-15)
        assert_allclose(transform(1, 0, 0, 0, 30), [1, 0, 0], atol=1e-15)
        assert_allclose(transform(1, 0, 0, 0, -20), [1, 0, 0], atol=1e-15)
        assert_allclose(transform(1, 0, 0, 0, -90), [1, 0, 0], atol=1e-15)

        # 4. when the direction of observation is at zero declination,
        #    an hour-angle of -6 hours (-90 degreees) makes W point to
        #    the east (positive X).
        assert_allclose(transform(1, 0, 0, -90, 0), [0, 0, 1], atol=1e-15)
        assert_allclose(transform(1, 0, 0, 90, 0), [0, 0, -1], atol=1e-15)

    def test_baselines(self):
        # There should be exactly N*(N-1)/2 baselines
        def test(ants_uvw):
            """

            :param ants_uvw:
            """
            bls = baselines(ants_uvw)
            l = len(ants_uvw)
            self.assertEqual(len(bls), l * (l - 1) // 2)

        for i in range(10):
            test(np.repeat(np.array(range(10 + i)), 3))

    def test_simulate_point(self):
        # Prepare a synthetic layout
        uvw = np.concatenate(np.concatenate(np.transpose(np.mgrid[-3:4, -3:4, 0:1])))
        bls = baselines(uvw)

        # Should have positive amplitude for the middle of the picture
        vis = simulate_point(bls, 0, 0)
        assert_allclose(vis, np.ones(len(vis)))

        # For the half-way point the result is either -1 or 1
        # depending on whether the baseline length is even
        bl_even = 1 - 2 * (numpy.sum(bls, axis=1) % 2)
        vis = simulate_point(bls, 0.5, 0.5)
        assert_allclose(vis, bl_even)
        vis = simulate_point(bls, -0.5, 0.5)
        assert_allclose(vis, bl_even)
        vis = simulate_point(bls, 0.5, -0.5)
        assert_allclose(vis, bl_even)
        vis = simulate_point(bls, -0.5, -0.5)
        assert_allclose(vis, bl_even)


if __name__ == '__main__':
    unittest.main()
