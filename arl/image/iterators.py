# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that define and manipulate images. Images are just data and a World Coordinate System.
"""

import logging

import numpy

from arl.image.operations import create_image_from_array

log = logging.getLogger(__name__)

def raster_iter(im, image_partitions=2, **kwargs):
    """Create a raster_iter generator, returning images

    The WCS is adjusted appropriately for each raster element. Hence this is a coordinate-aware
    way to iterate through an image.

    Provided we don't break reference semantics, memory should be conserved

    To update the image in place:
        for r in raster(im, image_parititions=2)::
            r.data[...] = numpy.sqrt(r.data[...])

    :param image_partitions: Number of image partitions on each axis (2)
    """

    log.info("raster: predicting using %d x %d image partitions" % (image_partitions, image_partitions))
    assert image_partitions <= im.nheight, "Cannot have more raster elements than pixels"
    assert image_partitions <= im.nwidth, "Cannot have more raster elements than pixels"
    assert im.nheight % image_partitions == 0, "The partitions must exactly fill the image"
    assert im.nwidth % image_partitions == 0, "The partitions must exactly fill the image"

    dx = int(im.nwidth // image_partitions)
    dy = int(im.nheight // image_partitions)
    log.info('raster: spacing of raster (%d, %d)' % (dx, dy))

    for y in range(0,im.nheight, dy):
        for x in range(0,im.nwidth, dx):
            log.debug('raster: partition (%d, %d) of (%d, %d)' %
                     (x//dx, y//dy, image_partitions, image_partitions))

            # Adjust WCS
            wcs = im.wcs.deepcopy()
            wcs.wcs.crpix[0] -= x
            wcs.wcs.crpix[1] -= y

            # Yield image from slice (reference!)
            yield create_image_from_array(im.data[..., y:y+dy, x:x+dx], wcs)
