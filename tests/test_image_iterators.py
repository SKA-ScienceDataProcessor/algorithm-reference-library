"""Unit tests for image operations

realtimcornwell@gmail.com
"""
import sys
import unittest

from arl.image.image_iterators import *
from arl.util.testing_support import create_test_image

log = logging.getLogger("tests.TestImageIterators")

class TestImageIterators(unittest.TestCase):

    def setUp(self):
        self.m31image = create_test_image()

    def test_rasterise(self):
    
        m31model=create_test_image(npol=4)
        nraster = 4
        for patch in raster_iter(m31model, nraster=nraster):
            assert patch.data.shape[3] == (m31model.data.shape[3] // nraster), \
                "Number of pixels in each patch: %d not as expected: %d" % (patch.data.shape[3],
                                                                        (m31model.data.shape[3] // nraster))
            assert patch.data.shape[2] == (m31model.data.shape[2] // nraster), \
                "Number of pixels in each patch: %d not as expected: %d" % (patch.data.shape[2],
                                                                        (m31model.data.shape[2] // nraster))
            
if __name__ == '__main__':
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    unittest.main()
