# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that define and manipulate images. Images are just data and a World Coordinate System.
"""

import logging

from arl.image.operations import create_image_from_array

log = logging.getLogger("arl.image_iterators")

class raster_iter:
    """Create a raster_iter generator, returning images

    The WCS is adjusted appropriately for each raster element. Hence this is a coordinate-aware
    way to iterate through an image.

    Provided we don't break reference semantics, memory should be conserved
    """
    
    def __init__(self, im, nraster=1):
        """Create a raster_iter generator, returning images
        
        The WCS is adjusted appropriately
    
        To update the image in place:
            for r in raster(im, nraster=2)::
                r.data[...] = numpy.sqrt(r.data[...])
        """
        assert nraster <= im.data.shape[3], "Cannot have more raster elements than pixels"
        assert nraster <= im.data.shape[2], "Cannot have more raster elements than pixels"
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
            log.info('image_iterators.raster: partition %d (%d, %d) of %d' % (self.location, x, y,
                                                                              self.nraster*self.nraster))
            x *= int(self.dx)
            y *= int(self.dy)
            sl = (..., slice(y, y + self.dy), slice(x, x + self.dx))
            self.location += 1
            # We should be able to use a slice on the wcs but it fails.
            wcs = self.im.wcs.deepcopy()
            wcs.wcs.crpix[0] -= x
            wcs.wcs.crpix[1] -= y
            return create_image_from_array(self.im.data[sl], wcs)
        else:
            raise StopIteration

