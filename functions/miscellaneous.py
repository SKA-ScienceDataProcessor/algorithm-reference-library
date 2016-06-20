# Tim Cornwell <realtimcornwell@gmail.com>
#
# The bulk of functions that have more than one data structure
#

import numpy
#

from astropy import wcs
from astropy import constants as const
from astropy.coordinates import SkyCoord, EarthLocation

from functions.visibility import visibility, visibility_from_configuration
from functions.configuration import configuration_filter, configuration_from_name
from functions.image import image, image_from_fits, image_from_array
from functions.skymodel import skymodel, skymodel_from_image
from functions.gaintable import gaintable
from functions.component import component

from crocodile.clean import *
from crocodile.synthesis import wcachefwd, wcachefwd
from crocodile.simulate import *


def wcs_from_visibility(vt: visibility, **kwargs):
    """
    Make a world coordinate system from the keyword args, setting defaults
    from the vistable
    """
    print("image: parsing kwargs to get definition of WCS")
    phasecentre = kwargs.get("phasecentre", vt.direction)
    reffrequency = kwargs.get("reffrequency", vt.frequency[0])
    channelwidth = kwargs.get("channelwidth", vt.frequency[1] - vt.frequency[0])

    npixel = kwargs.get("npixel", 1024)
    uvmax = 2.0 * numpy.sqrt((vt.data['uvw'][0] ** 2 + vt.data['uvw'][1] ** 2).max())
    cellsize = kwargs.get("cellsize", 1.0 / uvmax)

    shape=[npixel, npixel, len(vt.frequency), 4]

    w = wcs.WCS(naxis=4)
    w.wcs.cdelt = [-cellsize, cellsize, 1.0, channelwidth]
    w.wcs.crpix = [npixel // 2, npixel // 2, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.value, phasecentre.dec.value, 1.0, reffrequency]

    predfn = lambda *x: wcachefwd(*x, wstep=wstep, wcache=wcache)
    imgfn = lambda *x: wcacheimg(*x, wstep=wstep, wcache=wcache)

    d, p = doimg(npixel*cellsize, frequency[0]/const.c , vt.data['uvw'], vt.data['vis'], imgfn=imgfn)

    return shape, w


def invert(vt: visibility, **kwargs) -> (image, image):
    """
    Invert to make dirty image and PSF
    """
    shape, w = wcs_from_visibility(vt, **kwargs)
    dirty=image_from_array(numpy.zeros(shape), wcs)
    psf=image_from_array(numpy.zeros(shape), wcs)

    print("Stubbed: Inverting vistable to make dirty and psf")
    return (dirty, psf)


def predictl(vt: visibility, sm: skymodel, **kwargs) -> visibility:
    """
    Predict the visibility from a skymodel
    :type vis: visibility
    """
    print("Stubbed: Predicting vistable from sky model")
    return vt


def calibrate(vis: visibility, sm: skymodel, **kwargs) -> gaintable:
    """
    Calibrate using a sky model
    """
    print("Stubbed: Solving for gaintable")
    return gaintable()


def correct(vt: visibility, gt: gaintable, **kwargs) -> visibility:
    """
    Correct a vistable using a gaintable
    """
    print("Stubbed: Applying gaintab")
    return vt


def majorcycle(vt: visibility, sm: skymodel, **kwargs) -> (visibility, skymodel):
    """
    Perform major cycles
    """

    print("Stubbed: Performing %d major cycles" % kwargs.get('nmajor', 100))
    return vt, sm


def visibilitysum(vt: visibility, direction: SkyCoord, **kwargs) -> component:
    """
    Direct Fourier summation
    """

    print("Stubbed: Performing Direct Fourier Summation in direction %s")
    return component()


def fitcomponent(image: image, sm: skymodel, **kwargs) -> (skymodel, image):
    """
    Find components in image, return skymodel and residual image
    """
    print("Stubbed: Finding components in image, adding to skymodel")
    return sm, image


if __name__ == '__main__':
    kwargs = {}

    vlaa = configuration_filter(configuration_from_name('VLAA'), **kwargs)
    times = numpy.arange(-3.0, +3.0, 3.0 / 60.0) * numpy.pi / 12.0
    freq = numpy.arange(5e6, 150.0e6, 1e7)
    direction = SkyCoord('00h42m30s', '-41d12m00s', frame='icrs')
    vt = simulate(vlaa, times, freq, weight=1.0, direction=direction)
    print(vt.data)
    print(vt.frequency)
    m31image = image_from_fits("../data/models/m31.model.fits")
    m31sm = skymodel_from_image(m31image)
    vtpred = visibility_from_configuration(vlaa, times, freq, weight=1.0, direction=direction)
    vtpred = visibilityle_from_skymodel(vtpred, m31sm, **kwargs)
    dirty, psf = image_from_visibility(vtpred, **kwargs)
    m31smnew, res = component_fit_image(dirty, m31sm, **kwargs)

    print("Now to construct a wcs")
    shape, w = wcs_from_visibility(vt, **kwargs)
    print(shape)
    print(w)

    dirty, psf = image_from_visibility(vtpred)