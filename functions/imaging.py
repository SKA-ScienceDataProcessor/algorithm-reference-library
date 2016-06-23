# Tim Cornwell <realtimcornwell@gmail.com>
#
# The imaging functions that have more than one data structure
#

import numpy
#
import pylru

from astropy import units as u
from astropy import wcs
from astropy import constants as const
from astropy.coordinates import SkyCoord

from functions.visibility import visibility, simulate
from functions.configuration import configuration_filter, named_configuration
from functions.image import image, image_from_fits, image_from_array, image_replicate, image_to_fits
from functions.skymodel import skymodel, skymodel_from_image
from functions.component import component

from crocodile.synthesis import wcacheimg, wcachefwd, wkernaf, doimg, dopredict


def wcs_from_visibility(vt: visibility, **kwargs):
    """
    Make a world coordinate system from the keyword args, setting defaults
    from the visibility
    """
    print("imaging.wcs_from_visibility: Parsing kwargs to get definition of WCS")
    phasecentre = kwargs.get("phasecentre", vt.direction)
    reffrequency = kwargs.get("reffrequency", numpy.average(vt.frequency)) * u.Hz
    deffaultbw = vt.frequency[0]
    if len(vt.frequency) > 1:
        deffaultbw = vt.frequency[1] - vt.frequency[0]
    channelwidth = kwargs.get("channelwidth", deffaultbw) * u.Hz
    print("imaging.wcs_from_visibility: Defining image at %s, frequency %s Hz, and bandwidth %s Hz" % (phasecentre,
                                                                                                       reffrequency,
                                                                                                       channelwidth))

    npixel = kwargs.get("npixel", 512)
    uvmax = (numpy.abs(vt.data['uvw']).max() * u.m * reffrequency / const.c).value
    cellsize = kwargs.get("cellsize", 1.0 / (2.0 * uvmax))
    npol = 1
    # Beware of python indexing order! wcs and the array have opposite ordering
    shape = [len(vt.frequency), npol, npixel, npixel]
    w = wcs.WCS(naxis=4)
    w.wcs.cdelt = [-cellsize, cellsize, 1.0, channelwidth.value]
    w.wcs.crpix = [npixel // 2, npixel // 2, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.value, phasecentre.dec.value, 1.0, reffrequency.value]
    w.naxis = 4

    return shape, uvmax, w


def invert(vt: visibility, **kwargs) -> (image, image):
    """
    Invert to make dirty image and PSF
    """
    print("imaging.invert: Inverting visibility to make dirty and psf")
    shape, uvmax, w = wcs_from_visibility(vt, **kwargs)

    npixel = shape[3]
    cellsize = kwargs.get("cellsize", abs(w.wcs.cdelt[0]))
    criticalcellsize = 1.0/(4.0*uvmax)
    if cellsize > criticalcellsize:
        print("Resetting cellsize %f to criticalcellsize %f" %(cellsize, criticalcellsize))
    theta = npixel * cellsize

    print("imaging.invert: Specified npixel=%d, specified cellsize = %f, FOV = %f" % (npixel, cellsize, theta))
    print("imaging.invert: Critical cellsize = %f" % (1.0 / (2.0 *uvmax)))

    wstep = kwargs.get("wstep", 10000.0)
    wcachesize = int(numpy.ceil(numpy.abs(vt.data['uvw'][:, 2]).max() / wstep))
    print("imaging.invert: Making w-kernel cache of %d kernels" % (wcachesize))
    wcache = pylru.FunctionCacheManager(lambda iw: wkernaf(N=256, theta=theta, w=iw * wstep, s=15, Qpx=4), 10000)
    imgfn = lambda *x: wcacheimg(*x, wstep=wstep, wcache=wcache)

    d = numpy.zeros(shape)
    p = numpy.zeros(shape)

    pmax = 0.0
    for channel in range(len(vt.frequency)):
        for pol in range(1):
            print('imaging.invert: Inverting channel %d, polarisation %d' % (channel, pol))
            d[channel, pol, :, :], p[channel, 0, :, :], pmax = \
            doimg(theta, 1.0 / cellsize, vt.data['uvw'] * (vt.frequency[channel] / const.c).value,
                  vt.data['vis'][:, channel, pol], imgfn=imgfn)
        assert pmax > 0.0, ("No data gridded for channel %d" % channel)

    dirty = image_from_array(d, w)
    psf = image_from_array(p, w)
    print("imaging.invert: Finished making dirty and psf")
    return (dirty, psf, pmax)


def predict(vt: visibility, sm: skymodel, **kwargs) -> visibility:
    """
    Predict the visibility from a skymodel
    :type vis: visibility
    """
    print("imaging.predict: Predicting visibility from sky model")
    shape, uvmax, wvis = wcs_from_visibility(vt, **kwargs)

    wimage = sm.images[0].wcs

    shape = sm.images[0].data.shape
    npixel = shape[3]
    reffrequency = kwargs.get("reffrequency", numpy.average(vt.frequency)) * u.Hz
    cellsize = abs(wimage.wcs.cdelt[0])
    criticalcellsize = 1.0/(4.0*uvmax)
    theta = npixel * cellsize

    print("imaging.predict: Image npixel=%d, Image cellsize = %f, Image FOV = %f" % (npixel, cellsize, theta))
    print("imaging.predict: Critical cellsize = %f" % criticalcellsize)
    assert (cellsize <= criticalcellsize), "Image cellsize is above critical"

    wstep = kwargs.get("wstep", 10000.0)
    wcachesize = int(numpy.ceil(numpy.abs(vt.data['uvw'][:, 2]).max() / wstep))
    print("imaging.predict: Making w-kernel cache of %d kernels" % (wcachesize))
    wcache = pylru.FunctionCacheManager(lambda iw: wkernaf(N=256, theta=theta, w=iw * wstep, s=15, Qpx=4), 10000)
    predfn = lambda *x: wcachefwd(*x, wstep=wstep, wcache=wcache)

    for channel in range(len(vt.frequency)):
        for pol in range(1):
            print('imaging.invert: Predicting channel %d, polarisation %d' % (channel, pol))
            puvw, vt.data['vis'][:, channel, pol] = dopredict(theta, 1.0/cellsize, vt.data['uvw'] *
                                                        (vt.frequency[channel] / const.c).value,
                                                        sm.images[0].data[channel, pol, :, :], predfn=predfn)
    print("imaging.predict: Finished predicting visibility from sky model")
    return vt


def majorcycle(vt: visibility, sm: skymodel, **kwargs) -> (visibility, skymodel):
    """
    Perform major cycles
    """

    print("imaging.majorcycle: Stubbed: Performing %d major cycles" % kwargs.get('nmajor', 100))
    return vt, sm


def visibilitysum(vt: visibility, direction: SkyCoord, **kwargs) -> component:
    """
    Direct Fourier summation
    """

    print("imaging.visibilitysum: Stubbed: Performing Direct Fourier Summation in direction %s")
    return component()


def fitcomponent(image: image, sm: skymodel, **kwargs) -> (skymodel, image):
    """
    Find components in image, return skymodel and residual image
    """
    print("imaging.fitcomponent: Stubbed: Finding components in image, adding to skymodel")
    return sm, image


if __name__ == '__main__':
    import os
    os.chdir('../')
    print(os.getcwd())
    kwargs = {'wstep':100}

    vlaa = configuration_filter(named_configuration('VLAA'), **kwargs)
    vlaa.data['xyz']*=1.0/30.0
    times = numpy.arange(-3.0, +3.0, 3.0 / 60.0) * numpy.pi / 12.0
    frequency = numpy.arange(1.0e8, 1.50e8, 1e7)
    direction = SkyCoord('00h42m30s', '-41d12m00s', frame='icrs')
    vt = simulate(vlaa, times, frequency, weight=1.0, direction=direction)
    print(vt.frequency)
    m31image = image_from_fits("./data/models/m31.MOD")
    print("Max, min in m31 image = %.6f, %.6f" % (m31image.data.max(), m31image.data.min()))
    m31image = image_replicate(m31image, shape=[1, 1, 1, len(frequency)])
    print("Max, min in m31 image = %.6f, %.6f" % (m31image.data.max(), m31image.data.min()))
    print(m31image.data.shape)
    m31sm = skymodel_from_image(m31image)
    vtpred = simulate(vlaa, times, frequency, weight=1.0, direction=direction)
    vtpred = predict(vtpred, m31sm, **kwargs)
    print(numpy.max(numpy.abs(vtpred.data['vis'])))
    dirty, psf, sumwt = invert(vtpred, **kwargs)
    print(dirty.wcs)
    print("Max, min in dirty image = %.6f, %.6f, sum of weights = %f" % (dirty.data.max(), dirty.data.min(), sumwt))
    image_to_fits(dirty, 'dirty.fits')
    image_to_fits(psf, 'psf.fits')
    m31smnew, res = fitcomponent(dirty, m31sm, **kwargs)
