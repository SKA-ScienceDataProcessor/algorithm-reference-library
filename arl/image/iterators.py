#
"""
Functions that define and manipulate images. Images are just data and a World Coordinate System.
"""

import logging

import numpy

from arl.data.data_models import Image
from arl.data.parameters import get_parameter
from arl.image.operations import create_image_from_array

log = logging.getLogger(__name__)


def image_null_iter(im: Image, **kwargs) -> numpy.ndarray:
    """One time iterator

    :param im:
    :param kwargs:
    :return:
    """
    yield im


def image_raster_iter(im: Image, **kwargs) -> Image:
    """Create an image_raster_iter generator, returning images, optionally with overlaps

    The WCS is adjusted appropriately for each raster element. Hence this is a coordinate-aware
    way to iterate through an image.

    Provided we don't break reference semantics, memory should be conserved

    To update the image in place:
        for r in raster(im, facets=2)::
            r.data[...] = numpy.sqrt(r.data[...])

    :param im: Image
    :param facets: Number of image partitions on each axis (2)
    :param overlap: overlap in pixels
    :param kwargs: throw away unwanted parameters
    """
    
    nchan, npol, ny, nx = im.shape
    facets = get_parameter(kwargs, "facets", 1)
    overlap = get_parameter(kwargs, 'overlap', 0)
    log.debug("raster_overlap: predicting using %d x %d image partitions" % (facets, facets))
    assert facets <= ny, "Cannot have more raster elements than pixels"
    assert facets <= nx, "Cannot have more raster elements than pixels"
    
    sx = int((nx // facets))
    sy = int((ny // facets))
    dx = int((nx // facets) + 2 * overlap)
    dy = int((ny // facets) + 2 * overlap)

    log.debug('raster_overlap: spacing of raster (%d, %d)' % (dx, dy))
    
    for fy in range(facets):
        y = ny // 2 + sy * (fy - facets // 2) - overlap
        for fx in range(facets):
            x = nx // 2 + sx * (fx - facets // 2) - overlap
            if (x >= 0) and (x + dx) <= nx and (y >= 0) and (y + dy) <= ny:
                log.debug('raster_overlap: partition (%d, %d) of (%d, %d)' % (fy, fx, facets, facets))
                # Adjust WCS
                wcs = im.wcs.deepcopy()
                wcs.wcs.crpix[0] -= x
                wcs.wcs.crpix[1] -= y
                # yield image from slice (reference!)
                yield create_image_from_array(im.data[..., y:y + dy, x:x + dx], wcs, im.polarisation_frame)


def image_channel_iter(im: Image, subimages=1) -> Image:
    """Create a image_channel_iter generator, returning images

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
        if i + 1 < len(channels):
            channel_max = channels[i + 1]
        else:
            channel_max = nchan
        
        # Adjust WCS
        wcs = im.wcs.deepcopy()
        wcs.wcs.crpix[3] -= channel
        
        # Yield image from slice (reference!)
        yield create_image_from_array(im.data[channel:channel_max, ...], wcs, im.polarisation_frame)
