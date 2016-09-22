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
from arl.parameters import get_parameter

import logging
log = logging.getLogger( "arl.image_operations" )

def create_image_from_array(data: numpy.array, wcs: WCS = None) -> Image:
    """ Create an image from an array
    
    :param data:
    :type numpy.array:
    :param wcs:
    :type WCS:
    :returns: Image
    """
    fim = Image()
    fim.data = data
    fim.wcs = wcs
    return fim


def export_image_to_fits(im: Image, fitsfile: str = 'imaging.fits'):
    """ Write an image to fits
    
    :param im: Image
    :type Image:
    :param fitsfile: Name of output fits file
    :type str:
    """
    return fits.writeto(filename=fitsfile, data=im.data, header=im.wcs.to_header(), clobber=True)


def import_image_from_fits(fitsfile: str):
    """ Read an Image from fits
    
    :param fitsfile:
    :type str:
    :returns: Image
    """
    # Deal with relative file names in a consistent way
    if fitsfile[0] == '.':
        import os
        chome = os.environ['CROCODILE']
        fitsfile = "%s/%s" % (chome, fitsfile)
    hdulist = fits.open(fitsfile)
    fim = Image()
    fim.data = hdulist[0].data
    fim.wcs = WCS(fitsfile)
    hdulist.close()
    log.debug("import_image_from_fits: Max, min in %s = %.6f, %.6f" % (fitsfile, fim.data.max(), fim.data.min()))
    return fim


def add_wcs_to_image(im: Image, wcs: WCS):
    """ Add a WCS to an Image
    
    :param im:
    :type Image:
    :param wcs:
    :type WCS:
    :returns: Image
    """
    im.wcs = wcs.deepcopy()
    return im


def reproject_image(im: Image, newwcs: WCS, shape=None):
    """ Re-project an image to a new coordinate system
    
    Currently uses the reproject python package.
    TODO: Write tailored reproject routine

    :param im: Image to be reprojected
    :type Image:
    :param newwcs: New WCS
    :type WCS:
    :returns: Reprojected Image, Footprint Image
    """
    rep, foot = reproject_interp((im.data, im.wcs), newwcs, shape, order='bicubic',
                                 independent_celestial_slices=False)
    return create_image_from_array(rep, newwcs), create_image_from_array(foot, newwcs)


def fft_image(im: Image, params={}):
    """ FFT an image

    :param im:
    :type Image:
    :returns: Image
    """
    log.error("fft_image: not yet implemented")
    
    return im


def add_image(im1: Image, im2: Image, checkwcs=False):
    """ Add two images
    
    :param im1:
    :type Image:
    :param im2:
    :type Image:
    :param checkwcs: Check if the WCS agree.
    :type bool:
    :returns: Image
    """
    assert not checkwcs, "Checking WCS not yet implemented"
    return create_image_from_array(im1.data + im2.data, im1.wcs)


def aq_image(im, params={}):
    """Assess the quality of an image

    :param im:
    :type Image:
    :returns: QA
    """
    log.error("aq_image: not yet implemented")
    return QA()


def show_image(im: Image, fig=None, title: str = ''):
    """ Show an Image with coordinates using matplotlib

    :param im:
    :type Image:
    :param fig:
    :type Matplotlib.pyplot.figure:
    :param title:
    :type str:
    :returns:
    """
    if not fig:
        fig = plt.figure()
    fig.add_subplot(111, projection=im.wcs.sub(['longitude', 'latitude']))
    plt.clf()
    if len(im.data.shape) == 4:
        log.debug(im.data[0, 0, :, :])
        plt.imshow(im.data[0, 0, :, :], origin='lower', cmap='rainbow')
    elif len(im.data.shape) == 2:
        plt.imshow(im.data[:, :], origin='lower', cmap='rainbow')
    plt.xlabel('RA---SIN')
    plt.ylabel('DEC--SIN')
    plt.title(title)
    plt.colorbar()
    return fig
