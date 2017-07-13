"""Unit tests for pipelines


"""

import unittest

from arl.data.parameters import get_parameter

import logging

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
            assert get_parameter(None, 'foo', 'bar') == 'bar'
        
        kwargs = self.parameters
        t1(**kwargs)


if __name__ == '__main__':
    unittest.main()
