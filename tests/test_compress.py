"""Unit tests for Fourier transforms

realtimcornwell@gmail.com
"""
import logging
import os
import unittest

import numpy

import matplotlib.pyplot as plt

from astropy.convolution import Gaussian2DKernel, convolve
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord
from numpy.testing import assert_allclose

from arl.fourier_transforms.ftprocessor import *
from arl.fourier_transforms.compress import *
from arl.image.operations import export_image_to_fits
from arl.skymodel.operations import create_skycomponent, find_skycomponents, find_nearest_component, \
    insert_skycomponent
from arl.util.testing_support import create_named_configuration, run_unittests
from arl.visibility.operations import create_visibility, sum_visibility

log = logging.getLogger(__name__)


class TestCompress(unittest.TestCase):

    def setUp(self):
        self.params = {'npixel': 128,
                       'npol': 1,
                       'cellsize': 0.0018,
                       'reffrequency': 1e8}
    
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = numpy.arange(- numpy.pi / 4.0, 1.001 * numpy.pi / 4.0, numpy.pi / 16.0)
        self.frequency = numpy.array([1e8])
    
        self.reffrequency = numpy.max(self.frequency)
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox=2000.0)
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                              phasecentre=self.phasecentre, **self.params)
        self.model = create_image_from_visibility(self.vis, **self.params)

    def test_compress_decompress_grid_vis(self):
        """Test compression"""
        cvis = compress_visibility(self.vis, self.model)
        dvis = decompress_visibility(cvis, self.vis, self.model)
        assert dvis.nvis == self.vis.nvis
        
if __name__ == '__main__':
    run_unittests()
