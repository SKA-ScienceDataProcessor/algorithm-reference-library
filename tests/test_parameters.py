"""Unit tests for parameters

realtimcornwell@gmail.com
"""

import sys
import logging
import unittest

from arl.data.parameters import *
from arl.util.testing_support import create_test_image, run_unittests
log = logging.getLogger(__name__)

class TestParameters(unittest.TestCase):
    def setUp(self):
        self.parameters = {'npixel': 256, 'cellsize':0.1, 'spectral_mode':'mfs'}

    def test_getparameter(self):
    
        def t1(**kwargs):
            assert get_parameter(kwargs, 'cellsize') == 0.1
            assert get_parameter(kwargs, 'spectral_mode', 'channels') == 'mfs'
            assert get_parameter(kwargs, 'null_mode', 'mfs') == 'mfs'
            assert get_parameter(kwargs, 'foo', 'bar') == 'bar'
            assert get_parameter(kwargs, 'foo') is None
        
        kwargs = self.parameters
        t1(**kwargs)


if __name__ == '__main__':
    run_unittests()
