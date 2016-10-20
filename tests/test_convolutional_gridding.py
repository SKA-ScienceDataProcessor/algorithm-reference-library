import unittest
import itertools
import numpy
from numpy.testing import assert_allclose

from arl.convolutional_gridding import _w_kernel_function, _kernel_oversample, \
    _coordinates2, anti_aliasing_function
from arl.fft_support import *


class TestConvolutionalGridding(unittest.TestCase):
    
    def assertAlmostEqualScalar(self, a, result=1.0):
        w = result * numpy.ones_like(result)
        
    def _test_pattern(self, npixel):
        return _coordinates2(npixel)[0] + _coordinates2(npixel)[1] * 1j
    
    def test_anti_aliasing(self):
        for shape in [(4, 4), (5, 5), (4, 6), (7, 3)]:
            aaf = anti_aliasing_function(shape, 0, 10)
            self.assertEqual(aaf.shape, shape)
            self.assertAlmostEqual(aaf[shape[0] // 2, shape[1] // 2], 1)
    
    def test_w_kernel_function(self):
        assert_allclose(_w_kernel_function(5, 0.1, 0), 1.0)
        self.assertAlmostEqualScalar(_w_kernel_function(5, 0.1, 100)[2, 2], 1)
        self.assertAlmostEqualScalar(_w_kernel_function(10, 0.1, 100)[5, 5], 1)
        self.assertAlmostEqualScalar(_w_kernel_function(11, 0.1, 1000)[5, 5], 1)
    
    def test_kernel_oversampled_subgrid(self):
        # Oversampling should produce the same values where sub-grids overlap
        for npixel in range(3, 30):
            pat = self._test_pattern(npixel)
            kern = _kernel_oversample(pat, npixel, 1, npixel - 2)
            kern2 = _kernel_oversample(pat, npixel, 2, npixel - 2)
            assert_allclose(kern[0, 0], kern2[0, 0], atol=1e-15)
            kern3 = _kernel_oversample(pat, npixel, 3, npixel - 2)
            assert_allclose(kern[0, 0], kern3[0, 0], atol=1e-15)
            kern4 = _kernel_oversample(pat, npixel, 4, npixel - 2)
            for ux, uy in itertools.product(range(2), range(2)):
                assert_allclose(kern2[uy, ux], kern4[2 * uy, 2 * ux], atol=1e-15)
            kern8 = _kernel_oversample(pat, npixel, 8, npixel - 2)
            for ux, uy in itertools.product(range(3), range(3)):
                assert_allclose(kern4[uy, ux], kern8[2 * uy, 2 * ux], atol=1e-15)
    
    def test_kernel_scale(self):
        # Scaling the grid should not make a difference
        npixel = 10
        wff = numpy.zeros((npixel, npixel))
        wff[npixel // 2, npixel // 2] = 1  # Not the most interesting kernel...
        k = _kernel_oversample(wff, npixel, 1, npixel)
        k2 = _kernel_oversample(4 * wff, npixel * 2, 1, npixel)
        assert_allclose(k, k2)
    
    def test_w_kernel_normalisation(self):
        # Test w-kernel normalisation. This isn't quite perfect.
        # TODO: Address very poor normalisation.
        for kernel_oversampling in [4, 5, 6]:
            for npixel in [3, 5, 9, 16, 20, 24, 32, 64]:
                k = _kernel_oversample(_w_kernel_function(npixel + 2, 0.1, npixel * 10), npixel + 2,
                                       kernel_oversampling, npixel)
                assert_allclose(numpy.sum(k), kernel_oversampling ** 2,
                                rtol=0.07)
    
    def test_convolutional_grid(self):
        # Test for
        pass
    
    def test_convolutional_degrid(self):
        pass


if __name__ == '__main__':
    unittest.main()
