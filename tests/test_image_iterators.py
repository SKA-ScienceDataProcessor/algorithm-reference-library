"""Unit tests for image operations

realtimcornwell@gmail.com
"""
import unittest

import os
import sys
import numpy
from numpy.testing import assert_allclose

from arl.image_iterators import *
from arl.testing_support import create_test_image

import logging
log = logging.getLogger("tests.TestImageIterators")

class TestImageIterators(unittest.TestCase):

    def setUp(self):
        self.m31image = create_test_image()

    def test_rasterise(self):
    
        m31model=create_test_image()
        for patch in raster(m31model, nraster=2):
            pass

if __name__ == '__main__':
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    unittest.main()
