
import unittest
import itertools
import numpy as np
from numpy.testing import assert_allclose

from arl.synthesis_support import *
from arl.convolutional_gridding import _w_kernel_function, _kernel_oversample, _coordinates2
from arl.fft_support import *

class TestConvolutionalGridding(unittest.TestCase):
    
    def _pattern(self, N):
        return _coordinates2(N)[0]+_coordinates2(N)[1]*1j

    def test_anti_aliasing(self):
        for shape in [(4, 4), (5, 5), (4, 6), (7, 3)]:
            aaf = anti_aliasing_function(shape, 0, 10)
            self.assertEqual(aaf.shape, shape)
            self.assertAlmostEqual(aaf[shape[0] // 2, shape[1] // 2], 1)
    
    def test_w_kernel_function(self):
        assert_allclose(_w_kernel_function(5, 0.1, 0), 1)
        self.assertAlmostEqual(_w_kernel_function(5, 0.1, 100)[2, 2], 1)
        self.assertAlmostEqual(_w_kernel_function(10, 0.1, 100)[5, 5], 1)
        self.assertAlmostEqual(_w_kernel_function(11, 0.1, 1000)[5, 5], 1)
    
    def test_kernel_oversampled_subgrid(self):
        # Oversampling should produce the same values where sub-grids overlap
        for N in range(3, 30):
            pat = self._pattern(N)
            kern = _kernel_oversample(pat, N, 1, N - 2)
            kern2 = _kernel_oversample(pat, N, 2, N - 2)
            assert_allclose(kern[0, 0], kern2[0, 0], atol=1e-15)
            kern3 = _kernel_oversample(pat, N, 3, N - 2)
            assert_allclose(kern[0, 0], kern3[0, 0], atol=1e-15)
            kern4 = _kernel_oversample(pat, N, 4, N - 2)
            for ux, uy in itertools.product(range(2), range(2)):
                assert_allclose(kern2[uy, ux], kern4[2 * uy, 2 * ux], atol=1e-15)
            kern8 = _kernel_oversample(pat, N, 8, N - 2)
            for ux, uy in itertools.product(range(3), range(3)):
                assert_allclose(kern4[uy, ux], kern8[2 * uy, 2 * ux], atol=1e-15)
    
    def test_kernel_scale(self):
        # Scaling the grid should not make a difference
        N = 10
        wff = numpy.zeros((N, N))
        wff[N // 2, N // 2] = 1  # Not the most interesting kernel...
        k = _kernel_oversample(wff, N, 1, N)
        k2 = _kernel_oversample(wff, N * 2, 1, N)
        assert_allclose(k, k2 * 4)
    
    def test_w_kernel_normalisation(self):
        # Test w-kernel normalisation. This isn't quite perfect.
        for Qpx in [4, 5, 6]:
            for N in [3, 5, 9, 16, 20, 24, 32, 64]:
                k = _kernel_oversample(_w_kernel_function(N + 2, 0.1, N * 10), N + 2, Qpx, N)
                assert_allclose(numpy.sum(k), Qpx ** 2,
                                rtol=0.07)


if __name__ == '__main__':
    unittest.main()
