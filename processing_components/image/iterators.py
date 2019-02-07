#
"""
Functions that define and manipulate images. Images are just data and a World Coordinate System.
"""

import logging
import collections

import numpy

from data_models.memory_data_models import Image

from processing_library.image.operations import create_image_from_array, create_empty_image_like
from processing_library.util.array_functions import tukey_filter

log = logging.getLogger(__name__)


def image_null_iter(im: Image, facets=1, overlap=0) -> collections.Iterable:
    """One time iterator

    :param im:
    :param facets: Number of image partitions on each axis (2)
    :param overlap: overlap in pixels
    :return:
    """
    yield im


def image_raster_iter(im: Image, facets=1, overlap=0, taper='flat', make_flat=False) -> collections.Iterable:
    """Create an image_raster_iter generator, returning images, optionally with overlaps

    The WCS is adjusted appropriately for each raster element. Hence this is a coordinate-aware
    way to iterate through an image.

    Provided we don't break reference semantics, memory should be conserved. However make_flat
    creates a new set of images and thus reference semantics dont hold.

    To update the image in place:
        for r in raster(im, facets=2)::
            r.data[...] = numpy.sqrt(r.data[...])
            
    If the overlap is greater than zero, we choose to keep all images the same size so the
    other ring of facets are ignored. So if facets=4 and overlap > 0 then the iterator returns
    (facets-2)**2 = 4 images.
    
    A taper is applied in the overlap regions. None implies a constant value, linear is a ramp, and
    quadratic is parabolic at the ends.

    :param im: Image
    :param facets: Number of image partitions on each axis (2)
    :param overlap: overlap in pixels
    :param taper: method of tapering at the edges: 'flat' or 'linear' or 'quadratic' or 'tukey'
    :param make_flat: Make the flat images
    """
    nchan, npol, ny, nx = im.shape
    assert facets <= ny, "Cannot have more raster elements than pixels"
    assert facets <= nx, "Cannot have more raster elements than pixels"
    
    assert facets >=1, "Facets cannot be zero or less"
    assert overlap >= 0, "Overlap must be zero or greater"
    
    if facets == 1:
        yield im
    else:
        
        assert overlap < (nx // facets), "Overlap in facets is too large"
        assert overlap < (ny // facets), "Overlap in facets is too large"

        # Step between facets
        sx = nx // facets + overlap
        sy = ny // facets + overlap
    
        # Size of facet
        dx = sx + overlap
        dy = sy + overlap

        # Step between facets
        sx = nx // facets + overlap
        sy = ny // facets + overlap

        # Size of facet
        dx = nx // facets + 2 * overlap
        dy = nx // facets + 2 * overlap

        def taper_linear():
            t = numpy.ones(dx)
            ramp = numpy.arange(0, overlap).astype(float) / float(overlap)
            
            t[:overlap] = ramp
            t[(dx - overlap):dx] = 1.0 - ramp
            result = numpy.outer(t, t)
            
            return result

        def taper_quadratic():
            t = numpy.ones(dx)
            ramp = numpy.arange(0, overlap).astype(float) / float(overlap)
            
            quadratic_ramp = numpy.ones(overlap)
            quadratic_ramp[0:overlap // 2] = 2.0 * ramp[0:overlap // 2] ** 2
            quadratic_ramp[overlap // 2:] = 1 - 2.0 * ramp[overlap // 2:0:-1] ** 2
            
            t[:overlap] = quadratic_ramp
            t[(dx - overlap):dx] = 1.0 - quadratic_ramp
            
            result = numpy.outer(t, t)
            return result

        def taper_tukey():

            xs = numpy.arange(dx) / float(dx)
            r = 2 * overlap / dx
            t = [tukey_filter(x, r) for x in xs]
    
            result = numpy.outer(t, t)
            return result

        i = 0
        for fy in range(facets):
            y = ny // 2 + sy * (fy - facets // 2) - overlap // 2
            for fx in range(facets):
                x = nx // 2 + sx * (fx - facets // 2) - overlap // 2
                if (x >= 0) and (x + dx) <= nx and (y >= 0) and (y + dy) <= ny:
                    # Adjust WCS
                    wcs = im.wcs.deepcopy()
                    wcs.wcs.crpix[0] -= x
                    wcs.wcs.crpix[1] -= y
                    # yield image from slice (reference!)
                    subim = create_image_from_array(im.data[..., y:y + dy, x:x + dx], wcs, im.polarisation_frame)
                    if overlap > 0 and make_flat:
                        flat = create_empty_image_like(subim)
                        if taper == 'linear':
                            flat.data[..., :, :] = taper_linear()
                        elif taper == 'quadratic':
                            flat.data[..., :, :] = taper_quadratic()
                        elif taper == 'tukey':
                            flat.data[..., :, :] = taper_tukey()
                        else:
                            flat.data[...] = 1.0
                        yield flat
                    else:
                        yield subim
                    i += 1


def image_channel_iter(im: Image, subimages=1) -> collections.Iterable:
    """Create a image_channel_iter generator, returning images

    The WCS is adjusted appropriately for each raster element. Hence this is a coordinate-aware
    way to iterate through an image.

    Provided we don't break reference semantics, memory should be conserved

    To update the image in place:
        for r in raster(im, facets=2)::
            r.data[...] = numpy.sqrt(r.data[...])

    :param im: Image
    :param subimages: Number of subimages
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
