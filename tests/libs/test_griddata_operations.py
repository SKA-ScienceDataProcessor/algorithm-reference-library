""" Unit tests for image operations


"""
import logging
import unittest

import numpy

from processing_components.griddata.operations import create_griddata_from_image, fft_griddata_to_image, \
    fft_image_to_griddata, convert_griddata_to_image
from processing_components.simulation.testing_support import create_test_image

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
    
    def test_fftim(self):
        self.m31image = create_test_image(cellsize=0.001, frequency=[1e8], canonical=True)
        m31_griddata = fft_image_to_griddata(self.m31image)
        m31_fft_ifft = fft_griddata_to_image(m31_griddata)
        numpy.testing.assert_array_almost_equal(self.m31image.data, m31_fft_ifft.data.real, 12)


if __name__ == '__main__':
    unittest.main()
