import unittest

from arl.fourier_transforms.fft_support import *
from numpy.testing import assert_allclose

from util.convolutional_gridding import _coordinates2
from util.coordinate_support import *

class TestFFTSupport(unittest.TestCase):
    
    def _pattern(self, npixel):
        return _coordinates2(npixel)[0] + _coordinates2(npixel)[1] * 1j
    
    def test_pad_extract(self):
        for npixel, N2 in [(1, 1), (1, 2), (2, 3), (3, 4), (2, 5), (4, 6)]:
            cs = 1 + self._pattern(npixel)
            cs_pad = pad_mid(cs, N2)
            cs2 = 1 + self._pattern(N2) * N2 / npixel
            # At this point all fields in cs2 and cs_pad should either
            # be equal or zero.
            equal = numpy.abs(cs_pad - cs2) < 1e-15
            zero = numpy.abs(cs_pad) < 1e-15
            assert (equal + zero).all()
            # And extracting the middle should recover the original data
            assert_allclose(extract_mid(cs_pad, npixel), cs)
    
    def test_extract_oversampled(self):
        for npixel, kernel_oversampling in [(1, 2), (2, 3), (3, 2), (4, 2), (5, 3)]:
            a = 1 + self._pattern(npixel * kernel_oversampling)
            ex = extract_oversampled(a, 0, 0, kernel_oversampling, npixel) / kernel_oversampling ** 2
            assert_allclose(ex, 1 + self._pattern(npixel))

if __name__ == '__main__':
    unittest.main()
