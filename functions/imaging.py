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

from functions.visibility import Visibility, simulate, phaserotate
from functions.configuration import configuration_filter, named_configuration
from functions.image import Image, image_from_fits, image_from_array, image_replicate, image_to_fits
from functions.skymodel import SkyModel, skymodel_from_image, skymodel_add_component, skymodel_from_component
from functions.skycomponent import SkyComponent
from astropy.coordinates import SkyCoord, CartesianRepresentation

from crocodile.synthesis import wcacheimg, wcachefwd, wkernaf, doimg, dopredict
from crocodile.simulate import simulate_point


def wcs_from_visibility(vt: Visibility, **kwargs):
    """
    Make a world coordinate system from the keyword args, setting defaults
    from the Visibility
    """
    print("imaging.wcs_from_visibility: Parsing kwargs to get definition of WCS")
    imagecentre = kwargs.get("imagecentre", vt.phasecentre)
    phasecentre = kwargs.get("phasecentre", vt.phasecentre)
    reffrequency = kwargs.get("reffrequency", numpy.average(vt.frequency)) * u.Hz
    deffaultbw = vt.frequency[0]
    if len(vt.frequency) > 1:
        deffaultbw = vt.frequency[1] - vt.frequency[0]
    channelwidth = kwargs.get("channelwidth", deffaultbw) * u.Hz
    print("imaging.wcs_from_visibility: Defining Image at %s, frequency %s Hz, and bandwidth %s Hz" % (imagecentre,
                                                                                                       reffrequency,
                                                                                                       channelwidth))

    npixel = kwargs.get("npixel", 512)
    uvmax = (numpy.abs(vt.data['uvw']).max() * u.m * reffrequency / const.c).value
    cellsize = kwargs.get("cellsize", 1.0 / (2.0 * uvmax))
    npol = 4
    # Beware of python indexing order! wcs and the array have opposite ordering
    shape = [len(vt.frequency), npol, npixel, npixel]
    w = wcs.WCS(naxis=4)
    w.wcs.cdelt = [-cellsize, cellsize, 1.0, channelwidth.value]
    w.wcs.crpix = [npixel // 2, npixel // 2, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.value, phasecentre.dec.value, 1.0, reffrequency.value]
    w.naxis = 4

    return shape, uvmax, w, imagecentre


def invert(vt: Visibility, **kwargs) -> (Image, Image):
    """
    Invert to make dirty Image and PSF
    """
    print("imaging.invert: Inverting Visibility to make dirty and psf")
    shape, uvmax, w, imagecentre = wcs_from_visibility(vt, **kwargs)

    npixel = shape[3]
    cellsize = kwargs.get("cellsize", abs(w.wcs.cdelt[0]))
    criticalcellsize = 1.0 / (4.0 * uvmax)
    if cellsize > criticalcellsize:
        print("Resetting cellsize %f to criticalcellsize %f" % (cellsize, criticalcellsize))
    theta = npixel * cellsize

    print("imaging.invert: Specified npixel=%d, specified cellsize = %f, FOV = %f" % (npixel, cellsize, theta))
    print("imaging.invert: Critical cellsize = %f" % (1.0 / (2.0 * uvmax)))

    wstep = kwargs.get("wstep", 10000.0)
    wcachesize = int(numpy.ceil(numpy.abs(vt.data['uvw'][:, 2]).max() / wstep))
    print("imaging.invert: Making w-kernel cache of %d kernels" % (wcachesize))
    wcache = pylru.FunctionCacheManager(lambda iw: wkernaf(N=256, theta=theta, w=iw * wstep, s=15, Qpx=4), 10000)
    imgfn = lambda *x: wcacheimg(*x, wstep=wstep, wcache=wcache)

    # Apply a phase rotation from the visibility phase centre to the image phase centre
#    vtphaserotate = phaserotate(vt, imagecentre)

    d = numpy.zeros(shape)
    p = numpy.zeros(shape)

    pmax = 0.0
    nchan = shape[0]
    npol = shape[1]
    for channel in range(nchan):
        for pol in range(npol):
            print('imaging.invert: Inverting channel %d, polarisation %d' % (channel, pol))
            d[channel, pol, :, :], p[channel, 0, :, :], pmax = \
                doimg(theta, 1.0 / cellsize, vt.data['uvw'] *
                      (vt.frequency[channel] / const.c).value,
                      vt.data['vis'][:, channel, pol], imgfn=imgfn)
        assert pmax > 0.0, ("No data gridded for channel %d" % channel)

    dirty = image_from_array(d, w)
    psf = image_from_array(p, w)
    print("imaging.invert: Finished making dirty and psf")
    return (dirty, psf, pmax)


def predict(vt: Visibility, sm: SkyModel, **kwargs) -> Visibility:
    """
    Predict the visibility from a SkyModel
    :type vis: Visibility
    """
    vshape = vt.data['vis'].shape
    shape, uvmax, wvis, imagecentre = wcs_from_visibility(vt, **kwargs)

    if len(sm.images):
        print("imaging.predict: Predicting Visibility from sky model images")

        for im in sm.images:
            wimage = im.wcs

            vshape = vt.data['vis'].shape

            ishape = sm.images[0].data.shape
            npixel = ishape[3]
            nchan = ishape[0]
            npol = ishape[1]

            assert ishape[1] == vshape[2], "Image %d and visibility %d have different number of polarisations" % (
                ishape[1], vshape[2])
            assert ishape[0] == len(vt.frequency), "Image %d and visibility %d have different number of channels" % \
                                              (ishape[0], len(vt.frequency))
            cellsize = abs(wimage.wcs.cdelt[0])
            criticalcellsize = 1.0 / (4.0 * uvmax)
            theta = npixel * cellsize

            print("imaging.predict: Image npixel=%d, Image cellsize = %f, Image FOV = %f" % (npixel, cellsize, theta))
            print("imaging.predict: Critical cellsize = %f" % criticalcellsize)
            assert (cellsize <= criticalcellsize), "Image cellsize is above critical"

            wstep = kwargs.get("wstep", 10000.0)
            wcachesize = int(numpy.ceil(numpy.abs(vt.data['uvw'][:, 2]).max() / wstep))
            print("imaging.predict: Making w-kernel cache of %d kernels" % (wcachesize))
            wcache = pylru.FunctionCacheManager(lambda iw: wkernaf(N=256, theta=theta, w=iw * wstep, s=15, Qpx=4), 10000)
            predfn = lambda *x: wcachefwd(*x, wstep=wstep, wcache=wcache)

            for channel in range(nchan):
                uvw = vt.data['uvw'] * (vt.frequency[channel] / const.c).value
                for pol in range(npol):
                    print('imaging.predict: Predicting from image channel %d, polarisation %d' % (channel, pol))
                    puvw, vt.data['vis'][:, channel, pol] = dopredict(theta, 1.0 / cellsize, uvw,
                                                                      sm.images[0].data[channel, pol, :, :], predfn=predfn)
            print("imaging.predict: Finished predicting Visibility from sky model images")

    vdc = vt.phasecentre.represent_as(CartesianRepresentation)

    if len(sm.components):
        print("imaging.predict: Predicting Visibility from sky model components")

        for icomp in range(len(sm.components)):
            comp = sm.components[icomp]
            cshape = comp.flux.shape
            nchan = cshape[0]
            npol = cshape[1]

            assert cshape[1] == vshape[2], "Component %d and visibility %d have different number of polarisations" % (
                cshape[1], vshape[2])
            assert cshape[0] == len(vt.frequency), "Component %d and visibility %d have different number of channels" % \
                                              (cshape[0], len(vt.frequency))
            dc = comp.direction.represent_as(CartesianRepresentation)
            print('imaging.predict: Cartesian representation of component %d = (%f, %f)' % (icomp, dc.y, dc.z))
            for channel in range(nchan):
                uvw = vt.data['uvw'] * (vt.frequency[channel] / const.c).value
                # Calculate the phasor needed to rotate from the visibility tracking centre
                # to the component direction
                phasor = simulate_point(uvw, dc.y, dc.z)
                for pol in range(npol):
                    print('imaging.predict: Predicting from component %d channel %d, polarisation %d' % (icomp, channel,
                                                                                                         pol))
                    vt.data['vis'][:, channel, pol] = vt.data['vis'][:, channel, pol] + \
                                                      comp.flux[channel,pol] * numpy.conj(phasor)
        print("imaging.predict: Finished predicting Visibility from sky model components")

    return vt


def majorcycle(vt: Visibility, sm: SkyModel, cleaner, **kwargs) -> (Visibility, SkyModel):
    """
    Perform major cycles
    """
    nmajor = kwargs.get('nmajor', 5)
    print("imaging.majorcycle: Performing %d major cycles" % nmajor)

    # The model is added to each major cycle and then the visibilities are
    # calculated from the full model
    dirty, psf, sumwt = invert(vt, **kwargs)
    comp = Image(dirty)
    comp.data = 0.0 * dirty.data.copy()
    for i in range(nmajor):
        print("Start of major cycle %d" % (i))
        cc, res = cleaner(dirty, psf, **kwargs)
        comp += cc
        vtpred = predict(vt, sm, **kwargs)
        vt.data['vis'] = vtobs.data['vis]'] - vtpred.data['vis']
        dirty, psf, sumwt = invert(vt, **kwargs)
        if numpy.abs(dirty.data).max() < 1.1 * thresh:
            print("Reached stopping threshold %.6f Jy" % thresh)
            break
        print("End of major cycle")
    print("End of major cycles")
    return vt, sm

def fitcomponent(im: Image, **kwargs) -> SkyComponent:
    """
    Find components in Image, return SkyComponent, just find the peak for now
    """
    # TODO: Implement full image fitting of components
    print("imaging.fitcomponent: Finding components in Image")

    # Beware: The index sequencing is opposite in wcs and Python!
    locpeak = numpy.unravel_index((numpy.abs(im.data)).argmax(), im.data.shape)
    print("imaging.fitcomponent: Peak is at pixel coordinates %s" % str(locpeak))
    w = im.wcs.sub(['longitude', 'latitude'])
    worldpeak = w.wcs_pix2world(locpeak[3], locpeak[2], 1)
    sc=SkyCoord(worldpeak[0]*u.deg, worldpeak[1]*u.deg)
    print("imaging.fitcomponent: Peak is at %s" % sc)
    flux=im.data[:,:,locpeak[2],locpeak[3]]
    print("imaging.fitcomponent: Flux is %s" % flux)
    w = im.wcs.sub(['spectral'])
    frequency = w.wcs_pix2world(range(im.data.shape[0]), 1)
    return SkyComponent(direction=sc, flux=flux, frequency=frequency, shape='point')


def findflux(im: Image, sc: SkyCoord, **kwargs) -> SkyComponent:
    """
    Find flux at a given direction, return SkyComponent
    """
    print("imaging.findflux: Finding flux at a given direction in Image")
    worldloc = [sc.ra, sc.dec]
    print("imaging.findflux: Extracting flux at coordinates %s" % str(sc))
    w = im.wcs.sub(['longitude', 'latitude'])
    pixloc = w.wcs_world2pix(worldloc[0], worldloc[1], 1)
    print(pixloc)
    flux=im.data[:,:,int(pixloc[1]+0.5),int(pixloc[1]+0.5)]
    print("imaging.findflux: Flux is %s" % flux)
    return SkyComponent(direction=sc, flux=flux, frequency=[], shape='point')
