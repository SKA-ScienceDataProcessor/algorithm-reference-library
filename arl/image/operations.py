# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that define and manipulate images. Images are just data and a World Coordinate System.
"""

from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

from arl.data.data_models import *
from arl.data.parameters import *

log = logging.getLogger("image.operations")


def create_image_from_slice(im, imslice):
    """Create image from an image using a numpy.slice
    
    """
    fim = Image()
    fim.data = im.data[imslice]
    fim.wcs = im.wcs(imslice)
    return fim


def create_image_from_array(data: numpy.array, wcs: WCS = None) -> Image:
    """ Create an image from an array

    :rtype: Image
    :param data:
    :param wcs:
    :returns: Image
    """
    fim = Image()
    fim.data = data
    fim.wcs = wcs.deepcopy()
    return fim


def create_empty_image_like(im: Image) -> Image:
    """ Create an image from an array

    :param im:
    :returns: Image
    """
    fim = Image()
    fim.data = numpy.zeros_like(im.data)
    if im.wcs is None:
        fim.wcs = None
    else:
        fim.wcs = im.wcs.deepcopy()
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
    log.info("import_image_from_fits: Max, min in %s = %.6f, %.6f" % (fitsfile, fim.data.max(), fim.data.min()))
    return fim


def add_wcs_to_image(im: Image, wcs: WCS):
    """ Add a WCS to an Image

    :param im:
    :param wcs:
    :returns: Image
    """
    im.wcs = wcs.deepcopy()
    return im


def reproject_image(im: Image, newwcs: WCS, shape=None, params=None):
    """ Re-project an image to a new coordinate system
    
    Currently uses the reproject python package. This seems to have some features do be careful using this method.
    For timeslice imaging I had to use griddata.

    :param shape:
    :param im: Image to be reprojected
    :param newwcs: New WCS
    :param params: Dictionary of parameters
    :returns: Reprojected Image, Footprint Image
    """
    
    log.debug("reproject_image: Converting SIN projection from parameters %s to %s" %
              (im.wcs.wcs.get_pv(), newwcs.wcs.get_pv()))
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


def qa_image(im, params=None):
    """Assess the quality of an image

    :param params:
    :param im:
    :returns: QA
    """
    data = {'max': numpy.max(im.data),
            'min': numpy.min(im.data),
            'rms': numpy.std(im.data),
            'sum': numpy.sum(im.data),
            'medianabs': numpy.median(numpy.abs(im.data)),
            'median': numpy.median(im.data)}
    qa = QA(origin="qa_image",
            data=data,
            context=get_parameter(params, 'context', ""))
    return qa


def show_image(im: Image, fig=None, title: str = '', pol=0, chan=0):
    """ Show an Image with coordinates using matplotlib

    :param im:
    :param fig:
    :param title:
    :returns:
    """
    
    import matplotlib.pyplot as plt
    
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
