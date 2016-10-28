# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that define and manipulate images. Images are just data and a World Coordinate System.
"""

import logging

from image.image_operations import create_image_from_array

log = logging.getLogger("arl.image_iterators")

class raster_iter:
    """Create a raster_iter generator, returning images

    The WCS is adjusted appropriately

    Provided we don't break reference semantics, memory should be conserved
    """
    
    def __init__(self, im, nraster=1):
        """Create a raster_iter generator, returning images
        
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
        """Returns an image that can be modified using references"""
        if self.location < self.nraster * self.nraster:
            x = int(self.location // self.nraster)
            y = int(self.location - x * self.nraster)
            x *= int(self.dx)
            y *= int(self.dy)
            sl = (..., slice(y, y + self.dy), slice(x, x + self.dx))
            self.location += 1
            # We should be able to use a slice on the wcs but it fails.
            wcs = self.im.wcs.deepcopy()
            wcs.wcs.crpix[0] -= x
            wcs.wcs.crpix[1] -= y
#            return create_image_from_array(self.im.data[...,y:(y + self.dy):1,x:(x + self.dx):1], wcs)
            return create_image_from_array(self.im.data[sl], wcs)
        else:
            raise StopIteration

