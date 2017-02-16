"""Unit tests for Gaussian fit

realtimcornwell@gmail.com
"""
import unittest

import numpy as np
from numpy import pi, r_

from arl.misc.gaussian_fit import *

import logging

log = logging.getLogger(__name__)


class TestGaussianFit(unittest.TestCase):

    def test_fit(self):
        # Create the gaussian data
        Xin, Yin = np.mgrid[0:201, 0:201]
        data = gaussian(3.0, 100.0, 100.0, 20.0, 40.0, pi/3.0)(Xin, Yin)
        
        params = fitgaussian(data)
        fit = gaussian(*params)
    
        (height, x, y, width_x, width_y, rotation) = params
        assert abs(height-3.0) < 1e-7
        assert abs(x-100.0) < 1e-7
        assert abs(y-100.0) < 1e-7
        assert abs(width_x-20.0) < 1e-7
        assert abs(width_y-40.0) < 1e-7
        assert abs(rotation-pi/3) < 1e-7