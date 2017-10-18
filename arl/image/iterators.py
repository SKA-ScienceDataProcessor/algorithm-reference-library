
#
"""
Functions that define and manipulate images. Images are just data and a World Coordinate System.
"""

import numpy

import logging
from arl.data.data_models import Image
from arl.image.operations import create_image_from_array

log = logging.getLogger(__name__)

def raster_iter(im: Image, facets=2, **kwargs) -> Image:
    """Create a raster_iter generator, returning images

    The WCS is adjusted appropriately for each raster element. Hence this is a coordinate-aware
    way to iterate through an image.

    Provided we don't break reference semantics, memory should be conserved

    To update the image in place:
        for r in raster(im, facets=2)::
            r.data[...] = numpy.sqrt(r.data[...])

    :param im: Image
    :param facets: Number of image partitions on each axis (2)
    """

    log.debug("raster: predicting using %d x %d image partitions" % (facets, facets))
    assert facets <= im.nheight, "Cannot have more raster elements than pixels"
    assert facets <= im.nwidth, "Cannot have more raster elements than pixels"
    assert im.nheight % facets == 0, "The partitions must exactly fill the image"
    assert im.nwidth % facets == 0, "The partitions must exactly fill the image"

    dx = int(im.nwidth // facets)
    dy = int(im.nheight // facets)
    log.debug('raster: spacing of raster (%d, %d)' % (dx, dy))

    for y in range(0,im.nheight, dy):
        for x in range(0,im.nwidth, dx):
            log.debug('raster: partition (%d, %d) of (%d, %d)' %
                     (x//dx, y//dy, facets, facets))

            # Adjust WCS
            wcs = im.wcs.deepcopy()
            wcs.wcs.crpix[0] -= x
            wcs.wcs.crpix[1] -= y

            # Yield image from slice (reference!)
            yield create_image_from_array(im.data[..., y:y+dy, x:x+dx], wcs)

def channel_iter(im: Image, subimages=1, **kwargs) -> Image:
    """Create a channel_iter generator, returning images

    The WCS is adjusted appropriately for each raster element. Hence this is a coordinate-aware
    way to iterate through an image.

    Provided we don't break reference semantics, memory should be conserved

    To update the image in place:
        for r in raster(im, facets=2)::
            r.data[...] = numpy.sqrt(r.data[...])

    :param im: Image
    :param channel_width: Number of image partitions on each axis (2)
    """

    nchan, npol, ny, nx = im.shape
    
    assert subimages <= nchan, "More subimages %d than channels %d" % (subimages, nchan)
    step = nchan // subimages
    channels = numpy.array(range(0, nchan, step), dtype='int')
    assert len(channels) == subimages, "subimages %d does not match length of channels %d" % (subimages, len(channels))

    for i, channel in enumerate(channels):
        if i+1 < len(channels):
            channel_max = channels[i+1]
        else:
            channel_max = nchan

        # Adjust WCS
        wcs = im.wcs.deepcopy()
        wcs.wcs.crpix[3] -= channel

        # Yield image from slice (reference!)
        yield create_image_from_array(im.data[channel:channel_max,...], wcs)
