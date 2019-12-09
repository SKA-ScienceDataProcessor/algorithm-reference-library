""" Unit tests for image operations


"""
import logging
import unittest

import numpy

from processing_components.griddata.operations import create_griddata_from_image, convert_griddata_to_image
from processing_components.simulation import create_test_image

log = logging.getLogger(__name__)


class TestGridData(unittest.TestCase):
    
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.m31image = create_test_image(cellsize=0.0001)
        self.cellsize = 180.0 * 0.0001 / numpy.pi
    
    def test_create_griddata_from_image(self):
        m31model_by_image = create_griddata_from_image(self.m31image)
        assert m31model_by_image.shape[0] == self.m31image.shape[0]
        assert m31model_by_image.shape[1] == self.m31image.shape[1]
        assert m31model_by_image.shape[3] == self.m31image.shape[2]
        assert m31model_by_image.shape[4] == self.m31image.shape[3]
    
    def test_convert_griddata_to_image(self):
        m31model_by_image = create_griddata_from_image(self.m31image)
        m31_converted = convert_griddata_to_image(m31model_by_image)
    
if __name__ == '__main__':
    unittest.main()
