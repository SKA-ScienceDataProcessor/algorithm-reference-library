# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that define and manipulate images. Images are just data and a World Coordinate System.
"""

import copy

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

from arl.data.data_models import *
from arl.data.parameters import *

log = logging.getLogger(__name__)


def image_sizeof(im: Image):
    """ Return size in GB
    """
    return im.size()


def create_image_from_slice(im, imslice):
    """Create image from an image using a numpy.slice
    
    Both data and wcs are  sliced
    
    :param im: Image to be sliced
    :param imslice: Slice to be applied
    :returns: Sliced image
    
    """
    fim = Image()
    fim.data = im.data[imslice]
    fim.wcs = im.wcs(imslice).deepcopy()
    if image_sizeof(im) > 1.0:
        log.debug(
            "create_image_from_slice: created image of shape %s, size %.3f (GB)" % (str(im.shape), image_sizeof(im)))
    return fim


def create_image_from_array(data: numpy.array, wcs: WCS = None) -> Image:
    """ Create an image from an array and optional wcs

    :rtype: Image
    :param data:
    :param wcs:
    :returns: Image
    """
    fim = Image()
    fim.data = data
    fim.wcs = wcs.deepcopy()
    if image_sizeof(fim):
        log.debug("create_image_from_array: created image of shape %s, size %.3f (GB)" % (str(fim.shape),
                                                                                          image_sizeof(fim)))
    return fim


def copy_image(im: Image) -> Image:
    """ Create an image from an array
    
    Performs deepcopy of data, breaking reference semantics

    :param im:
    :returns: Image
    """
    fim = Image()
    fim.data = copy.deepcopy(im.data)
    if im.wcs is None:
        fim.wcs = None
    else:
        fim.wcs = copy.deepcopy(im.wcs)
    if image_sizeof(fim) >= 1.0:
        log.debug("copy_image: created image of shape %s, size %.3f (GB)" % (str(fim.shape), image_sizeof(fim)))
    return fim


def create_empty_image_like(im: Image) -> Image:
    """ Create an empty image like another in shape and wcs

    :param im:
    :returns: Image
    """
    fim = Image()
    fim.data = numpy.zeros_like(im.data)
    if im.wcs is None:
        fim.wcs = None
    else:
        fim.wcs = copy.deepcopy(im.wcs)
    if image_sizeof(im) >= 1.0:
        log.debug("create_empty_image_like: created image of shape %s, size %.3f (GB)" % (str(im.shape),
                                                                                          image_sizeof(im)))
    return fim


def export_image_to_fits(im: Image, fitsfile: str = 'imaging.fits'):
    """ Write an image to fits
    
    :param im: Image
    :param fitsfile: Name of output fits file
    """
    return fits.writeto(filename=fitsfile, data=im.data, header=im.wcs.to_header(), overwrite=True)


def import_image_from_fits(fitsfile: str):
    """ Read an Image from fits
    
    :param fitsfile:
    :returns: Image
    """
    hdulist = fits.open(arl_path(fitsfile))
    fim = Image()
    fim.data = hdulist[0].data
    fim.wcs = WCS(arl_path(fitsfile))
    hdulist.close()
    log.info("import_image_from_fits: created image of shape %s, size %.3f (GB)" %
             (str(fim.shape), image_sizeof(fim)))
    log.info("import_image_from_fits: Max, min in %s = %.6f, %.6f" % (fitsfile, fim.data.max(), fim.data.min()))
    return fim


def reproject_image(im: Image, newwcs: WCS, shape=None):
    """ Re-project an image to a new coordinate system
    
    Currently uses the reproject python package. This seems to have some features do be careful using this method.
    For timeslice imaging I had to use griddata.


    :param im: Image to be reprojected
    :param newwcs: New WCS
    :param shape:
    :returns: Reprojected Image, Footprint Image
    """
    
    rep, foot = reproject_interp((im.data, im.wcs), newwcs, shape, order='bicubic',
                                 independent_celestial_slices=True)
    return create_image_from_array(rep, newwcs), create_image_from_array(foot, newwcs)


def checkwcs(wcs1, wcs2):
    """ Check for compatbility of wcs
    
    :param wcs1:
    :param wcs2:
    """
    assert wcs1.compare(wcs2, comp=wcs2.WCSCOMPARE_ANCILLARY), "WCS's do not agree"


def add_image(im1: Image, im2: Image, docheckwcs=False):
    """ Add two images
    
    :param docheckwcs:
    :param im1:
    :param im2:
    :returns: Image
    """
    if docheckwcs:
        checkwcs(im1.wcs, im2.wcs)
    
    return create_image_from_array(im1.data + im2.data, im1.wcs)


def qa_image(im, mask=None, **kwargs):
    """Assess the quality of an image

    :param params:
    :param im:
    :returns: QA
    """
    if mask is None:
        data = {'shape': str(im.data.shape),
                'max': numpy.max(im.data),
                'min': numpy.min(im.data),
                'rms': numpy.std(im.data),
                'sum': numpy.sum(im.data),
                'medianabs': numpy.median(numpy.abs(im.data)),
                'median': numpy.median(im.data)}
    else:
        mdata = im.data[mask.data > 0.0]
        data = {'shape': str(im.data.shape),
                'max': numpy.max(mdata),
                'min': numpy.min(mdata),
                'rms': numpy.std(mdata),
                'sum': numpy.sum(mdata),
                'medianabs': numpy.median(numpy.abs(mdata)),
                'median': numpy.median(mdata)}
    
    qa = QA(origin="qa_image",
            data=data,
            context=get_parameter(kwargs, 'context', ""))
    return qa


def show_image(im: Image, fig=None, title: str = '', pol=0, chan=0):
    """ Show an Image with coordinates using matplotlib

    :param im:
    :param fig:
    :param title:
    :returns:
    """
    
    if not fig:
        fig = plt.figure()
    plt.clf()
    fig.add_subplot(111, projection=im.wcs.sub(['longitude', 'latitude']))
    if len(im.data.shape) == 4:
        plt.imshow(numpy.real(im.data[chan, pol, :, :]), origin='lower', cmap='rainbow')
    elif len(im.data.shape) == 2:
        plt.imshow(numpy.real(im.data[:, :]), origin='lower', cmap='rainbow')
    plt.xlabel('RA---SIN')
    plt.ylabel('DEC--SIN')
    plt.title(title)
    plt.colorbar()
    return fig
