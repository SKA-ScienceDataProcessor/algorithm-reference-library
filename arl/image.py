# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

import matplotlib.pyplot as plt

import numpy as numpy

from astropy.wcs import WCS
from astropy.io import fits
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy.coordinates import SkyCoord
from arl.skycomponent import SkyComponent, create_skycomponent

"""
Functions that define and manipulate images. Images are just data and a World Coordinate System.
"""

class Image:
    """Image class with Image data (as a numpy.array) and optionally the AstroPy WCS.
    
    Many operations can be done conveniently using numpy arl on Image.data.
    
    Most of the imaging arl require an image in canonical format:
    - 4 axes: RA, DEC, POL, FREQ
    
    The conventions for indexing in WCS and numpy are opposite.
    - In astropy.wcs, the order is (longitude, latitude, polarisation, frequency)
    - in numpy, the order is (frequency, polarisation, latitude, longitude)
    
    """

    def __init__(self):
        self.data = None
        self.wcs = None


def image_show(im: Image, fig = None, title: str = ''):
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
    plt.imshow(im.data[0, 0, :, :], origin='lower', cmap='rainbow')
    plt.xlabel('RA---SIN')
    plt.ylabel('DEC--SIN')
    plt.title(title)
    plt.colorbar()
    return fig


def image_filter(im: Image, **kwargs):
    """ Filter an image

    :param im:
    :type Image:
    :param kwargs:
    :returns:
    """
    print("image_filter: No filter implemented yet")
    return im


def image_from_array(data: numpy.array, wcs: WCS = None) -> Image:
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


def image_to_fits(im: Image, fitsfile: str = 'imaging.fits'):
    """ Write an image to fits
    
    :param im: Image
    :type Image:
    :param fitsfile: Name of output fits file
    :type str:
    """
    return fits.writeto(fitsfile, im.data, im.wcs.to_header(), clobber=True)


def image_from_fits(fitsfile: str):
    """ Read an Image from fits
    
    :param fitsfile:
    :type str:
    :returns: Image
    """
    # Deal with relative file names in a consistent way
    if fitsfile[0] == '.':
        import os
        chome = os.environ['CROCODILE']
        fitsfile="%s/%s" % (chome, fitsfile)
    hdulist = fits.open(fitsfile)
    fim = Image()
    fim.data = hdulist[0].data
    fim.wcs = WCS(fitsfile)
    print("image_from_fits: Max, min in %s = %.6f, %.6f" % (fitsfile, fim.data.max(), fim.data.min()))
    return fim


def image_add_wcs(im: Image, wcs: WCS):
    """ Add a WCS to an Image
    
    :param im:
    :type Image:
    :param wcs:
    :type WCS:
    :returns: Image
    """
    im.wcs = wcs.deepcopy()
    return im


def image_replicate(im: Image, shape=None):
    """ Make a new canonical shape Image, extended along third and fourth axes by replication.
    
    The order is [chan, pol, dec, ra]


    :param im:
    :type Image:
    :param shape: Extra axes (only axes 0 and 1 are heeded.
    :type 4-sequence:
    :returns: Image
    """
    # TODO: Fill in extra axes in wcs
    if shape is None:
        shape = [1, 1, 1, 1]
    if len(im.data.shape) == 2:
        fim = Image()
        image_add_wcs(fim, im.wcs)
        fshape = [shape[3], shape[2], im.data.shape[1], im.data.shape[0]]
        fim.data = numpy.zeros(fshape)
        print("image_replicate: replicating shape %s to %s" % (im.data.shape, fim.data.shape))
        for i3 in range(shape[3]):
            for i2 in range(shape[2]):
                fim.data[i3, i2, :, :] = im.data[:, :]
        print(fim.wcs)
    elif len(im.data.shape) == 3:
        fim = Image()
        image_add_wcs(fim, im.wcs)
        # TODO: fix this for non-square images!
        fshape = [shape[3], im.data.shape[2], im.data.shape[1], im.data.shape[0]]
        fim.data = numpy.array(fshape)
        print("Image: replicating shape %s to %s" % (im.data.shape, fim.data.shape))
        for i3 in range(shape[3]):
            fim.data[i3, :, :, :] = im.data[:, :, :]
    else:
        fim = im

    return fim


def image_add(im1: Image, im2: Image, checkwcs=False):
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
    return image_from_array(im1.data + im2.data, im1.wcs)

def fitcomponent(im: Image, **kwargs) -> SkyComponent:
    """ Find components in Image, return SkyComponent, just find the peak for now
    
    :param im: Image to be searched
    :type Image:
    :returns: SkyComponent
    """
    # TODO: Implement full image fitting of components
    print("imaging.fitcomponent: Finding components in Image")

    # Beware: The index sequencing is opposite in wcs and Python!
    locpeak = numpy.unravel_index((numpy.abs(im.data)).argmax(), im.data.shape)
    print("imaging.fitcomponent: Found peak at pixel coordinates %s" % str(locpeak))
    w = im.wcs.sub(['longitude', 'latitude'])
    sc = pixel_to_skycoord(locpeak[3], locpeak[2], im.wcs, 0, 'wcs')
    print("imaging.fitcomponent: Found peak at world coordinates %s" % str(sc))
    flux=im.data[:,:,locpeak[2],locpeak[3]]
    print("imaging.fitcomponent: Flux is %s" % flux)
    # We also need the frequency values
    w = im.wcs.sub(['spectral'])
    frequency = w.wcs_pix2world(range(im.data.shape[0]), 1)
    return create_skycomponent(direction=sc, flux=flux, frequency=frequency, shape='point')


def findflux(im: Image, sc: SkyCoord, **kwargs) -> SkyComponent:
    """ Find flux at a given direction, return SkyComponent
    
    :param im:
    :type Image:
    :param sc:
    :type SkyCoord:
    :returns: SkyComponent
    
    """
    print("imaging.findflux: Extracting flux at world coordinates %s" % str(sc))
    w = im.wcs.sub(['longitude', 'latitude'])
    pixloc = skycoord_to_pixel(sc, im.wcs, 0, 'wcs')
    print("imaging.findflux: Extracting flux at pixel coordinates %d %d" % (pixloc[0], pixloc[1]))
    flux=im.data[:,:,int(pixloc[1]+0.5),int(pixloc[0]+0.5)]
    print("imaging.findflux: Flux is %s" % flux)

    # We also need the frequency values
    w = im.wcs.sub(['spectral'])
    frequency = w.wcs_pix2world(range(im.data.shape[0]), 0)

    return create_skycomponent(direction=sc, flux=flux, frequency=frequency, shape='point')

if __name__ == '__main__':
    import os
    chome=os.environ['CROCODILE']
    kwargs = {}
    m31model = image_from_fits("%s/data/models/M31.MOD" % chome)
    m31model_by_array = image_from_array(m31model.data, m31model.wcs)
    try:
        m31modelsum = image_filter(image_add(m31model, m31model_by_array, checkwcs=True), **kwargs)
    except:
        print("Image: correctly failed on checkwcs=True")
        pass
    m31modelsum = image_filter(image_add(m31model, m31model_by_array), **kwargs)
    print(m31model.data.shape)
    print(m31model.wcs)
    print(image_to_fits(m31model, fitsfile='temp.fits'))
