# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

import numpy

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS

from reproject import reproject_interp

from arl.data_models import *


"""
Functions that define and manipulate images. Images are just data and a World Coordinate System.
"""

def create_test_image(canonical=True):
    """Create a useful test image
    
    This is the test image M31 widely used in ALMA and other simulations. It is actually part of an Halpha region in
    M31.
    
    :param canonical: Make the image into a 4 dimensional image
    :returns: Image
    """
    chome = os.environ['CROCODILE']
    im = create_image_from_fits("%s/data/models/M31.MOD" % chome)
    if canonical:
        im=replicate_image(im)
    return im


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
        print(im.data[0, 0, :, :])
        plt.imshow(im.data[0, 0, :, :], origin='lower', cmap='rainbow')
    elif len(im.data.shape) == 2:
        plt.imshow(im.data[:, :], origin='lower', cmap='rainbow')
    plt.xlabel('RA---SIN')
    plt.ylabel('DEC--SIN')
    plt.title(title)
    plt.colorbar()
    return fig


def filter_image(im: Image, **kwargs):
    """ Filter an image

    :param im:
    :type Image:
    :param kwargs:
    :returns:
    """
    print("filter_image: No filter implemented yet")
    return im


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


def save_image_to_fits(im: Image, fitsfile: str = 'imaging.fits'):
    """ Write an image to fits
    
    :param im: Image
    :type Image:
    :param fitsfile: Name of output fits file
    :type str:
    """
    return fits.writeto(filename=fitsfile, data=im.data, header=im.wcs.to_header(), clobber=True)


def create_image_from_fits(fitsfile: str):
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
    print("create_image_from_fits: Max, min in %s = %.6f, %.6f" % (fitsfile, fim.data.max(), fim.data.min()))
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


def replicate_image(im: Image, shape=None, frequency=1.4e9):
    """ Make a new canonical shape Image, extended along third and fourth axes by replication.
    
    The order is [chan, pol, dec, ra]


    :param im:
    :type Image:
    :param shape: Extra axes (only axes 0 and 1 are heeded.
    :type 4-sequence:
    :returns: Image
    """
    if shape == None:
        shape=[1,1,1,1]

    if len(im.data.shape) == 2:
        fim = Image()

        newwcs = WCS(naxis=4)

        newwcs.wcs.crpix = [im.wcs.wcs.crpix[0], im.wcs.wcs.crpix[1], 1.0, 1.0]
        newwcs.wcs.cdelt = [im.wcs.wcs.cdelt[0], im.wcs.wcs.cdelt[1], 1.0, 1.0]
        newwcs.wcs.crval = [im.wcs.wcs.crval[0], im.wcs.wcs.crval[1], 1.0, frequency]
        newwcs.wcs.ctype = [im.wcs.wcs.ctype[0], im.wcs.wcs.ctype[1], 'STOKES', 'FREQ']

        add_wcs_to_image(fim, newwcs)
        fshape = [shape[3], shape[2], im.data.shape[1], im.data.shape[0]]
        fim.data = numpy.zeros(fshape)
        print("replicate_image: replicating shape %s to %s" % (im.data.shape, fim.data.shape))
        for i3 in range(shape[3]):
            for i2 in range(shape[2]):
                fim.data[i3, i2, :, :] = im.data[:, :]
    else:
        fim = im
    
    return fim


def reproject_image(im: Image, newwcs: WCS, shape=None):
    """ Reproject an image to a new coordinate system
    
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


def fft_image(im: Image, **kwargs):
    """ FFT an image

    :param im:
    :type Image:
    :returns: Image
    """
    print("fft_image: not yet implemented")
   
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


def aq_image(im, **kwargs):
    """Assess the quality of an image

    :param im:
    :type Image:
    :returns: AQ
    """
    print("assess_quality.aq_image: not yet implemented")
    return AQ()


if __name__ == '__main__':
    import os
    from arl.skymodel_operations import create_skycomponent

    chome = os.environ['CROCODILE']
    kwargs = {}
    m31model = create_image_from_fits("%s/data/models/M31.MOD" % chome)
    m31model_by_array = create_image_from_array(m31model.data, m31model.wcs)
    try:
        m31modelsum = filter_image(add_image(m31model, m31model_by_array, checkwcs=True), **kwargs)
    except:
        print("Image: correctly failed on checkwcs=True")
        pass
    m31modelsum = filter_image(add_image(m31model, m31model_by_array), **kwargs)
    print(m31model.data.shape)
    print(m31model.wcs)
    print(save_image_to_fits(m31model, fitsfile='temp.fits'))
