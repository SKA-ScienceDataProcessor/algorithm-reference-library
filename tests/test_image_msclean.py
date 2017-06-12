"""Unit tests for image deconvolution via MSClean


"""
import unittest
import numpy
import logging

from arl.image.cleaners import create_scalestack, convolve_scalestack, convolve_convolve_scalestack,\
    argmax

log = logging.getLogger(__name__)


class TestImageMSClean(unittest.TestCase):
    def setUp(self):
        self.npixel = 256
        self.scales = [0.0, 8.0/numpy.sqrt(2.0), 8.0]
        self.stackshape = [len(self.scales), self.npixel, self.npixel]
        self.scalestack = create_scalestack(self.stackshape, self.scales)
    
    def test_convolve(self):
        img = numpy.zeros([self.npixel, self.npixel])
        img[75, 31] = 1.0
        result = convolve_scalestack(self.scalestack, img)
        assert argmax(result)[1:] == (75, 31)
        numpy.testing.assert_array_almost_equal(result[0, 75, 31],
                                                self.scalestack[0, self.npixel // 2, self.npixel // 2], 7)
        numpy.testing.assert_array_almost_equal(result[1, 75, 31],
                                                self.scalestack[1, self.npixel // 2, self.npixel // 2], 7)
        numpy.testing.assert_array_almost_equal(result[2, 75, 31],
                                                self.scalestack[2, self.npixel // 2, self.npixel // 2], 7)
    
    def test_convolve_convolve(self):
        img = numpy.zeros([self.npixel, self.npixel])
        img[75, 31] = 1.0
        result = convolve_convolve_scalestack(self.scalestack, img)
        assert argmax(result)[2:] == (75, 31)
        numpy.testing.assert_array_almost_equal(result[0, 0, 75, 31], self.scalestack[0, self.npixel // 2,
                                                                                      self.npixel // 2], 7)
        numpy.testing.assert_array_almost_equal(result[0, 1, 75, 31], self.scalestack[1, self.npixel // 2,
                                                                                      self.npixel // 2], 7)
        numpy.testing.assert_array_almost_equal(result[0, 2, 75, 31], self.scalestack[2, self.npixel // 2,
                                                                                      self.npixel // 2], 7)
        # This is a coarse test since the scales do not having the property of widths adding incoherently under
        # convolution
        numpy.testing.assert_array_almost_equal(result[1, 1, 75, 31], self.scalestack[2, self.npixel // 2,
                                                                                      self.npixel // 2], 2)
