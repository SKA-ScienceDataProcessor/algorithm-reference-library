"""Unit tests for Fourier transforms

realtimcornwell@gmail.com
"""
import unittest
import numpy
import logging

from arl.util.array_functions import average_chunks

from arl.util.testing_support import run_unittests

log = logging.getLogger(__name__)


class TestArray_functions(unittest.TestCase):

    def test_average_chunks(self):
        
        arr = numpy.linspace(0.0, 100.0, 11)
        wts = numpy.ones_like(arr)
        carr, cwts = average_chunks(arr, wts, 2)
        assert len(carr) == len(cwts)
        answerarr = numpy.array([5., 25., 45., 65.0, 85.0, 100.0])
        answerwts = numpy.array([2.0, 2.0, 2.0, 2.0, 2.0, 1.0])
        numpy.testing.assert_array_equal(carr, answerarr)
        numpy.testing.assert_array_equal(cwts, answerwts)

if __name__ == '__main__':
    run_unittests()
