# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

from collections import namedtuple

import numpy as numpy

from astropy.wcs import WCS
from astropy.nddata import NDData
from astropy.io import fits


class image():
    """
    Image class with image data (as a numpy.array) and optionally WCS
    """
    def __init__(self):
        self.data = None
        self.wcs = None


def image_filter(fim: image, **kwargs):
    print("image: No filter implemented yet")
    return fim


def image_from_array(data: image, wcs: image = None) -> image:
    """
    :type image: numpy.array
    """
    fim = image()
    fim.data = data
    fim.wcs = wcs
    return fim


def image_from_fits(fitsfile: str):
    hdulist = fits.open(fitsfile)
    return image_from_array(hdulist[0].data, WCS(fitsfile))


def image_add_wcs(im: image, wcs: WCS):
    im.wcs = WCS
    return im


def image_add(im1: image, im2: image, checkwcs=False):
    assert not checkwcs, "Checking WCS not yet implemented"
    return image_from_array(im1.data + im2.data, im1.wcs)


if __name__ == '__main__':
    kwargs = {}
    m31model = image_from_fits("../data/models/m31.model.fits")
    m31model_by_array = image_from_array(m31model.data, m31model.wcs)
    try:
        m31modelsum = image_filter(image_add(m31model, m31model_by_array, checkwcs=True), **kwargs)
    except:
        print("image: correctly failed on checkwcs=True")
        pass
    m31modelsum = image_filter(image_add(m31model, m31model_by_array), **kwargs)
    print(m31model.data.shape)
    print(m31model.wcs)
