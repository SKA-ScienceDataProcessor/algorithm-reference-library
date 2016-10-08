
from arl.synthesis_support import *
from crocodile.simulate import *

import unittest
import itertools
import numpy as np
from numpy.testing import assert_allclose

from astropy.coordinates import SkyCoord
from astropy import units as u

class TestKernel(unittest.TestCase):

    def test_coordinates(self):
        for N in [4,5,6,7,8,9]:
            low, high = coordinateBounds(N)
            c = coordinates(N)
            cy, cx = coordinates2(N)
            self.assertAlmostEqual(np.min(c), low)
            self.assertAlmostEqual(np.max(c), high)
            self.assertAlmostEqual(np.min(cx), low)
            self.assertAlmostEqual(np.max(cx), high)
            self.assertAlmostEqual(np.min(cy), low)
            self.assertAlmostEqual(np.max(cy), high)
            self.assertAlmostEqual(c[N//2], 0)
            assert (cx[:,N//2] == 0).all()
            assert (cy[N//2,:] == 0).all()

    def _pattern(self, N):
        return coordinates2(N)[0]+coordinates2(N)[1]*1j

    def test_pad_extract(self):
        for N, N2 in [ (1,1), (1,2), (2,3), (3,4), (2,5), (4,6) ]:
            cs = 1 + self._pattern(N)
            cs_pad = pad_mid(cs, N2)
            cs2 = 1 + self._pattern(N2) * N2 / N
            # At this point all fields in cs2 and cs_pad should either
            # be equal or zero.
            equal = numpy.abs(cs_pad - cs2) < 1e-15
            zero = numpy.abs(cs_pad) < 1e-15
            assert numpy.all(equal + zero)
            # And extracting the middle should recover the original data
            assert_allclose(extract_mid(cs_pad, N), cs)

    def test_extract_oversampled(self):
        for N, Qpx in [ (1,2), (2,3), (3,2), (4,2), (5,3) ]:
            a = 1+self._pattern(N * Qpx)
            ex = extract_oversampled(a, 0, 0, Qpx, N)/Qpx**2
            assert_allclose(ex, 1+self._pattern(N))

    def test_anti_aliasing(self):
        for shape in [(4,4),(5,5),(4,6),(7,3)]:
            aaf = anti_aliasing_function(shape, 0, 10)
            self.assertEqual(aaf.shape, shape)
            self.assertAlmostEqual(aaf[shape[0]//2,shape[1]//2], 1)

    def test_w_kernel_function(self):
        assert_allclose(w_kernel_function(5,0.1,0), 1)
        self.assertAlmostEqual(w_kernel_function(5,0.1,100)[2,2], 1)
        self.assertAlmostEqual(w_kernel_function(10,0.1,100)[5,5], 1)
        self.assertAlmostEqual(w_kernel_function(11,0.1,1000)[5,5], 1)

    def test_kernel_oversampled_subgrid(self):
        # Oversampling should produce the same values where sub-grids overlap
        for N in range(3,30):
            pat = self._pattern(N)
            kern = kernel_oversample(pat, N, 1, N-2)
            kern2 = kernel_oversample(pat, N, 2, N-2)
            assert_allclose(kern[0,0], kern2[0,0], atol=1e-15)
            kern3 = kernel_oversample(pat, N, 3, N-2)
            assert_allclose(kern[0,0], kern3[0,0], atol=1e-15)
            kern4 = kernel_oversample(pat, N, 4, N-2)
            for ux, uy in itertools.product(range(2), range(2)):
                assert_allclose(kern2[uy,ux], kern4[2*uy,2*ux], atol=1e-15)
            kern8 = kernel_oversample(pat, N, 8, N-2)
            for ux, uy in itertools.product(range(3), range(3)):
                assert_allclose(kern4[uy,ux], kern8[2*uy,2*ux], atol=1e-15)

    def test_kernel_scale(self):
        # Scaling the grid should not make a difference
        N = 10
        wff = numpy.zeros((N,N))
        wff[N//2,N//2] = 1 # Not the most interesting kernel...
        k = kernel_oversample(wff, N, 1, N)
        k2 = kernel_oversample(wff, N*2, 1, N)
        assert_allclose(k, k2*4)

    def test_w_kernel_normalisation(self):
        # Test w-kernel normalisation. This isn't quite perfect.
        for Qpx in [4,5,6]:
            for N in [3,5,9,16,20,24,32,64]:
                k = kernel_oversample(w_kernel_function(N+2,0.1,N*10), N+2, Qpx, N)
                assert_allclose(numpy.sum(k), Qpx**2,
                                rtol=0.07)
if __name__ == '__main__':
    unittest.main()
