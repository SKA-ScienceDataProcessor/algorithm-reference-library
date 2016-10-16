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
from arl.parameters import *

import logging

log = logging.getLogger("arl.image_operations")

def create_image_from_slice(im, slice):
    """Create image from an image using a numpy.slice
    
    """
    fim = Image()
    fim.data = im.data[slice]
    fim.wcs = im.wcs(slice)
    return fim


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


def create_empty_image_like(im: Image) -> Image:
    """ Create an image from an array

    :param im:
    :type Image:
    :returns: Image
    """
    fim = Image()
    fim.data = numpy.zeros_like(im.data)
    if im.wcs == None:
        fim.wcs = None
    else:
        fim.wcs = im.wcs.deepcopy()
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
    hdulist = fits.open(crocodile_path(fitsfile))
    fim = Image()
    fim.data = hdulist[0].data
    fim.wcs = WCS(crocodile_path(fitsfile))
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



def reproject_image(im: Image, newwcs: WCS, shape=None, params={}):
    """ Re-project an image to a new coordinate system
    
    Currently uses the reproject python package.
    TODO: Write tailored reproject routine

    :param im: Image to be reprojected
    :type Image:
    :param newwcs: New WCS
    :type WCS:
    :param params: Dictionary of parameters
    :returns: Reprojected Image, Footprint Image
    """
    # TODO: implement
    
    
    log_parameters(params)
    rep, foot = reproject_interp((im.data, im.wcs), newwcs, shape, order='bicubic',
                                 independent_celestial_slices=False)
    return create_image_from_array(rep, newwcs), create_image_from_array(foot, newwcs)


def fft_image(im: Image, params={}):
    """ FFT an image

    :param im:
    :type Image:
    :returns: Image
    """
    # TODO: implement
    
    
    log_parameters(params)
    log.error("fft_image: not yet implemented")
    
    return im

def checkwcs(wcs1, wcs2):
    """ Check for compatbility of wcs
    
    :param wcs1:
    :param wcs2:
    """
    # TODO: implement checkwcs
    return True

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

    
    if checkwcs:
        assert not checkwcs(im1.wcs, im2.wcs), "Checking WCS not yet implemented"

    return create_image_from_array(im1.data + im2.data, im1.wcs)


def aq_image(im, params={}):
    """Assess the quality of an image

    :param im:
    :type Image:
    :returns: QA
    """
    # TODO: implement
     
    log.error("aq_image: not yet implemented")
    log_parameters(params)
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
        plt.imshow(im.data[0, 0, :, :], origin='lower', cmap='rainbow')
    elif len(im.data.shape) == 2:
        plt.imshow(im.data[:, :], origin='lower', cmap='rainbow')
    plt.xlabel('RA---SIN')
    plt.ylabel('DEC--SIN')
    plt.title(title)
    plt.colorbar()
    return fig
