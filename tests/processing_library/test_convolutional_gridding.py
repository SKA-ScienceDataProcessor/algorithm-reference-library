""" Unit processing_library for convolutional Gridding


"""
import random
import unittest

import numpy
from numpy.testing import assert_allclose

from processing_library.fourier_transforms.convolutional_gridding import w_beam, coordinates, \
    coordinates2, coordinateBounds


class TestConvolutionalGridding(unittest.TestCase):
    
    @staticmethod
    def assertAlmostEqualScalar(a, result=1.0):
        result * numpy.ones_like(result)

    def test_coordinates(self):
        for N in [4, 5, 6, 7, 8, 9, 1000, 1001, 1002, 1003]:
            low, high = coordinateBounds(N)
            c = coordinates(N)
            cx, cy = coordinates2(N)
            self.assertAlmostEqual(numpy.min(c), low)
            self.assertAlmostEqual(numpy.max(c), high)
            self.assertAlmostEqual(numpy.min(cx), low)
            self.assertAlmostEqual(numpy.max(cx), high)
            self.assertAlmostEqual(numpy.min(cy), low)
            self.assertAlmostEqual(numpy.max(cy), high)
            assert c[N // 2] == 0
            assert (cx[N // 2, :] == 0).all()
            assert (cy[:, N // 2] == 0).all()

    @staticmethod
    def _test_pattern(npixel):
        return coordinates2(npixel)[0] + coordinates2(npixel)[1] * 1j

    def test_w_kernel_beam(self):
        assert_allclose(numpy.real(w_beam(5, 0.1, 0))[0, 0], 1.0)
        self.assertAlmostEqualScalar(w_beam(5, 0.1, 100)[2, 2], 1)
        self.assertAlmostEqualScalar(w_beam(10, 0.1, 100)[5, 5], 1)
        self.assertAlmostEqualScalar(w_beam(11, 0.1, 1000)[5, 5], 1)

if __name__ == '__main__':
    unittest.main()
