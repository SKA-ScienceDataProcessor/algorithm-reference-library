import unittest

from numpy.testing import assert_allclose
from arl.fft_support import *
from crocodile.simulate import *

from arl.convolutional_gridding import _w_kernel_function, _kernel_oversample, _coordinates2

class TestSynthesis(unittest.TestCase):
    
    def _pattern(self, N):
        return _coordinates2(N)[0] + _coordinates2(N)[1] * 1j
    
    def test_pad_extract(self):
        for N, N2 in [(1, 1), (1, 2), (2, 3), (3, 4), (2, 5), (4, 6)]:
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
        for N, Qpx in [(1, 2), (2, 3), (3, 2), (4, 2), (5, 3)]:
            a = 1 + self._pattern(N * Qpx)
            ex = extract_oversampled(a, 0, 0, Qpx, N) / Qpx ** 2
            assert_allclose(ex, 1 + self._pattern(N))

if __name__ == '__main__':
    unittest.main()
