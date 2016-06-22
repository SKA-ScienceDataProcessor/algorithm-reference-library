# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

import matplotlib as plt

import numpy as numpy

from astropy.wcs import WCS
from astropy.io import fits


class image():
    """
    Image class with image data (as a numpy.array) and optionally WCS
    """
    def __init__(self):
        self.data = None
        self.wcs = None

def image_show(im:image, title: str = ''):

    plt.clf()
    fig = plt.figure()
    fig.add_subplot(111, projection=im.wcs.sub(['longitude', 'latitude']))
    plt.imshow(im.data[0, 0, :, :], origin='lower', cmap='rainbow')
    plt.xlabel('RA---SIN')
    plt.ylabel('DEC--SIN')
    plt.colorbar()
    plt.show()
    return fig


def image_filter(fim: image, **kwargs):
    print("image_filter: No filter implemented yet")
    return fim


def image_from_array(data: numpy.array, wcs: WCS = None) -> image:
    """
    :type image: numpy.array
    """
    fim = image()
    fim.data = data
    fim.wcs = wcs
    return fim


def image_to_fits(im: image, fitsfile: str = 'immaging.fits'):
    """
    Write an image to fits
    :type image: image
    :param fitsfile: Name of output fits file
    """
    return fits.writeto(fitsfile, im.data, im.wcs.to_header(), clobber=True)


def image_from_fits(fitsfile: str):
    """
    Read an image from fits
    :param fitsfile:
    :return:
    """
    hdulist = fits.open(fitsfile)
    fim=image()
    fim.data=hdulist[0].data
    fim.wcs=WCS(fitsfile)
    print("image_from_fits: Max, min in %s = %.6f, %.6f" % (fitsfile, fim.data.max(), fim.data.min()))
    return fim


def image_add_wcs(im: image, w: WCS):
    """
    Add a WCS to an image
    :param im:
    :param wcs:
    :return:
    """
    im.wcs = w.deepcopy()
    return im

def image_replicate(im: image, shape: [] = [1,1,1,1]):
    """
    Make a new canonical shape image, extended along third and fourth axes by replication. The order is
    [chan, pol, dec, ra]

    TODO: Fill in extra axes in wcs

    :param im:
    :param shape: Extra axes (only axes 0 and 1 are heeded.
    :return:
    """
    if len(im.data.shape)==2:
        fim = image()
        image_add_wcs(fim, im.wcs)
        fshape =[shape[3], shape[2], im.data.shape[1], im.data.shape[0]]
        fim.data=numpy.zeros(fshape)
        print("image_replicate: replicating shape %s to %s" % (im.data.shape, fim.data.shape))
        for i3 in range(shape[3]):
            for i2 in range(shape[2]):
                fim.data[i3, i2, :, :]=im.data[:, :]
        print(fim.wcs)
    elif len(im.data.shape) == 3:
        fim = image()
        image_add_wcs(fim, im.wcs)
        # TODO: fix this for non-square images!
        fshape =[shape[3], im.data.shape[2], im.data.shape[1], im.data.shape[0]]
        fim.data=numpy.array(fshape)
        print("image: replicating shape %s to %s" % (im.data.shape, fim.data.shape))
        for i3 in range(shape[3]):
            fim.data[i3, :, :, :] = im.data[:, :, :]

    else:
        fim = im

    return fim


def image_add(im1: image, im2: image, checkwcs=False):
    assert not checkwcs, "Checking WCS not yet implemented"
    return image_from_array(im1.data + im2.data, im1.wcs)


if __name__ == '__main__':
    kwargs = {}
    m31model = image_from_fits("./data/models/M31.MOD")
    m31model_by_array = image_from_array(m31model.data, m31model.wcs)
    try:
        m31modelsum = image_filter(image_add(m31model, m31model_by_array, checkwcs=True), **kwargs)
    except:
        print("image: correctly failed on checkwcs=True")
        pass
    m31modelsum = image_filter(image_add(m31model, m31model_by_array), **kwargs)
    print(m31model.data.shape)
    print(m31model.wcs)
    print(image_to_fits(m31model, fitsfile='temp.fits'))
