"""Unit tests for image deconvolution

realtimcornwell@gmail.com
"""
import unittest

import numpy
from numpy.testing import assert_allclose
from arl.image.image_deconvolution import *


class TestImageDeconvolution(unittest.TestCase):

    def setUp(self):
        pass

    @unittest.skip("No test for deconvolve and restore cube")
    def test_deconvolve_and_restore_cube(self):
        pass

    @unittest.skip("No test for deconvolve and restore MSMFS")
    def test_deconvolve_and_restore_MSMFS(self):
        pass
    
    
if __name__ == '__main__':
    unittest.main()
