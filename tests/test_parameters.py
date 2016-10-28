"""Unit tests for parameters

realtimcornwell@gmail.com
"""

import logging
import unittest

from data.parameters import *
log = logging.getLogger("arl.test_parameters")

class TestParameters(unittest.TestCase):
    def setUp(self):
        self.paramsfile = crocodile_path("tests/TestParameters.txt")
        self.parameters = {'npixel': 256, 'cellsize':0.1, 'predict':{'cellsize':0.2}, 'invert':{'spectral_mode':'mfs'}}

    def test_exportimport(self):
        log_parameters(self.parameters)
        export_parameters(self.parameters, self.paramsfile)
        d = import_parameters(self.paramsfile)
        log_parameters(d)
        assert d == self.parameters


    def test_getparameter(self):
    
        assert get_parameter(self.parameters, 'cellsize') == 0.1
        assert get_parameter(self.parameters['predict'], 'cellsize') == 0.2
        assert get_parameter(self.parameters, 'spectral_mode', 'channels') == 'channels'
        assert get_parameter(self.parameters['invert'], 'spectral_mode') == 'mfs'
        assert get_parameter(self.parameters, 'foo', 'bar') == 'bar'
        assert get_parameter(self.parameters, 'foo') is None


if __name__ == '__main__':
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    unittest.main()
