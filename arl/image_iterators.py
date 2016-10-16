# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that define and manipulate images. Images are just data and a World Coordinate System.
"""

import numpy

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

# from reproject import reproject_interp

from arl.data_models import *
from arl.parameters import *
from arl.image_operations import create_image_from_array

import logging

log = logging.getLogger("arl.image_operations")

class raster():
    
    def __init__(self, im, nraster=1):
        """Create a raster generator, returning images
        
        The WCS is adjusted appropriately
    
        Provided we don't break reference semantics, memory should be conserved
        """
        self.dx = int(im.data.shape[3] // nraster)
        self.dy = int(im.data.shape[2] // nraster)
        self.im = im
        self.nraster = nraster
        self.location = 0
    
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.location < self.nraster * self.nraster:
            x = int(self.location // self.nraster)
            y = int(self.location - x * self.nraster)
            x *= int(self.dx)
            y *= int(self.dy)
            sl = slice(..., slice(y, y + self.dy), slice(x, x + self.dx))
            self.location += 1
            # We should be able to use a slice on the wcs but it fails.
            wcs = self.im.wcs.deepcopy()
            wcs.wcs.crpix[0] -= x
            wcs.wcs.crpix[1] -= y
            return create_image_from_array(self.im.data[...,y:(y + self.dy):1,x:(x + self.dx):1], wcs)
        else:
            raise StopIteration

