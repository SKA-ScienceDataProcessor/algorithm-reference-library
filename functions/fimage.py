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


def fimage():
    """
    Image class with image data (as a numpy.array) and optionally WCS
    """
    fim = namedtuple('fimage', ['data', 'wcs'])
    fim.data = None
    fim.wcs = None
    return fim


def fimage_filter(fim: fimage, **kwargs):
    print("fimage: No filter implemented yet")
    return fim


def fimage_from_array(data: numpy.array, wcs: WCS = None):
    """
    :type image: numpy.array
    """
    fim = fimage()
    fim.data = data
    fim.wcs = wcs
    return fim


def fimage_from_fits(fitsfile: str):
    hdulist = fits.open(fitsfile)
    return fimage_from_array(hdulist[0].data, WCS(fitsfile))


def fimage_add_wcs(im: fimage, wcs: WCS):
    im.wcs = WCS
    return im


def fimage_add_fimage(im1: fimage, im2: fimage, checkwcs=False):
    assert not checkwcs, "Checking WCS not yet implemented"
    return fimage_from_array(im1.data + im2.data, im1.wcs)


if __name__ == '__main__':
    kwargs = {}
    m31model = fimage_from_fits("../data/models/m31.model.fits")
    m31model_by_array = fimage_from_array(m31model.data, m31model.wcs)
    try:
        m31modelsum = fimage_filter(fimage_add_fimage(m31model, m31model_by_array, checkwcs=True), **kwargs)
    except:
        print("fimage: correctly failed on checkwcs=True")
        pass
    m31modelsum = fimage_filter(fimage_add_fimage(m31model, m31model_by_array), **kwargs)
    print(m31model.data.shape)
    print(m31model.wcs)
