# Tim Cornwell <realtimcornwell@gmail.com>
#
# Synthesis imaging functions
#

import numpy
import pylru
from astropy import constants as const
from astropy import units as units
from astropy import wcs
from astropy.coordinates import CartesianRepresentation

from crocodile.simulate import simulate_point, skycoord_to_lmn
from crocodile.synthesis import wcacheimg, wcachefwd, wkernaf, doimg, dopredict

from arl.data_models import *
from arl.image_operations import create_image_from_array
from arl.parameters import get_parameter

import logging
log = logging.getLogger("arl.fourier_transforms")

"""
Functions that perform imaging i.e. conversion of an Image to/from a Visibility
"""


def create_wcs_from_visibility(vis: Visibility, params={}, level=1):
    """Make a world coordinate system from params and Visibility

    :param vis:
    :type Visibility: Visibility to be processed
    :param params: keyword=value parameters
    :param level: level in params 0 = toplevel, 1 = parent, 2 = parent of parent.
    :returns: WCS
    """
    log.debug("fourier_transformscreate_wcs_from_visibility: Parsing parameters to get definition of WCS")
    imagecentre = get_parameter(params, "imagecentre", vis.phasecentre, level)
    phasecentre = get_parameter(params, "phasecentre", vis.phasecentre, level)
    reffrequency = get_parameter(params, "reffrequency", numpy.max(vis.frequency), level) * units.Hz
    deffaultbw = vis.frequency[0]
    if len(vis.frequency) > 1:
        deffaultbw = vis.frequency[1] - vis.frequency[0]
    channelwidth = get_parameter(params, "channelwidth", deffaultbw) * units.Hz
    log.debug("fourier_transforms.create_wcs_from_visibility: Defining Image at %s, frequency %s, and bandwidth %s"
          % (imagecentre, reffrequency, channelwidth))

    npixel = get_parameter(params, "npixel", 512, level)
    uvmax = (numpy.abs(vis.data['uvw']).max() * reffrequency / const.c).value
    log.debug("fourier_transforms.create_wcs_from_visibility: uvmax = %f lambda" % uvmax)
    criticalcellsize = 1.0 / (uvmax * 2.0)
    log.debug("fourier_transforms.create_wcs_from_visibility: Critical cellsize = %f radians, %f degrees" % (
        criticalcellsize, criticalcellsize * 180.0 / numpy.pi))
    cellsize = get_parameter(params, "cellsize", 0.5 * criticalcellsize, level)
    log.debug("fourier_transforms.create_wcs_from_visibility: Cellsize          = %f radians, %f degrees" % (cellsize,
                                                                                       cellsize * 180.0 / numpy.pi))
    if cellsize > criticalcellsize:
        log.debug("Resetting cellsize %f radians to criticalcellsize %f radians" % (cellsize, criticalcellsize))
        cellsize = criticalcellsize

    npol = 4
    # Beware of python indexing order! wcs and the array have opposite ordering
    shape = [len(vis.frequency), npol, npixel, npixel]
    w = wcs.WCS(naxis=4)
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, 1.0, channelwidth.value]
    w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.value, phasecentre.dec.value, 1.0, reffrequency.value]
    w.naxis = 4

    w.wcs.radesys = get_parameter(params, 'frame', 'ICRS')
    w.wcs.equinox = get_parameter(params, 'equinox', 2000.0)

    return shape, reffrequency, cellsize, w, imagecentre


def invert_visibility(vis: Visibility, params={}):
    """Invert to make dirty Image and PSF

    :param vis:
    :type Visibility: Visibility to be processed
    :returns: (dirty image, psf)
    """
    log.debug("fourier_transforms.invert_visibility: Inverting Visibility to make dirty and psf")
    shape, reffrequency, cellsize, w, imagecentre = create_wcs_from_visibility(vis, params=params)

    npixel = shape[3]
    theta = npixel * cellsize

    log.debug("fourier_transforms.invert_visibility: Specified npixel=%d, cellsize = %f rad, FOV = %f rad" %
          (npixel, cellsize, theta))

    # Set up the gridding kernel. We try to use a cached version
    gridding_algorithm = get_parameter(params, 'gridding_algorithm', 'wprojection')

    if gridding_algorithm == 'wprojection':
        log.debug("fourier_transforms.invert_visibility: Gridding by w projection")

        wstep = get_parameter(params, "wstep", 10000.0)

        wcachesize = int(numpy.ceil(numpy.abs(vis.data['uvw'][:, 2]).max() * reffrequency.value / (const.c.value * wstep)))
        log.debug("fourier_transforms.invert_visibility: Making w-kernel cache of %d kernels" % wcachesize)
        wcache = pylru.FunctionCacheManager(lambda iw: wkernaf(N=256, theta=theta, w=iw * wstep, s=15, Qpx=4), 10000)
        imgfn = lambda *x: wcacheimg(*x, wstep=wstep, wcache=wcache)
    else:
        raise NotImplementedError("gridding algorithm %s not supported" % gridding_algorithm)

    # Apply a phase rotation from the visibility phase centre to the image phase centre
    #    visphaserotate = phaserotate(vis, imagecentre)

    d = numpy.zeros(shape)
    p = numpy.zeros(shape)

    spectral_mode = get_parameter(params, 'spectral_mode', 'channel')
    log.debug('fourier_transforms.invert_visibility: spectral mode is %s' % spectral_mode)

    if spectral_mode == 'channel':
        pmax = 0.0
        nchan = shape[0]
        npol = shape[1]
        for channel in range(nchan):
            for pol in range(npol):
                log.debug('fourier_transforms.invert_visibility: Inverting channel %d, polarisation %d' % (channel, pol))
                d[channel, pol, :, :], p[channel, 0, :, :], pmax = \
                    doimg(theta, 1.0 / cellsize, vis.data['uvw'] *
                          (vis.frequency[channel] / const.c).value,
                          vis.data['vis'][:, channel, pol], imgfn=imgfn)
            assert pmax > 0.0, ("No data gridded for channel %d" % channel)
    else:
        raise NotImplementedError("mode %s not supported" % spectral_mode)


    dirty = create_image_from_array(d, w)
    psf = create_image_from_array(p, w)
    log.debug("fourier_transforms.invert_visibility: Finished making dirty and psf")


    return dirty, psf, pmax


def predict_visibility(vis: Visibility, sm: SkyModel, params={}) -> Visibility:
    """Predict the visibility (in place) from a SkyModel including both components and images

    :param vis:
    :type Visibility: Visibility to be processed
    :param sm:
    :type SkyModel:
    :returns: Visibility
    """
    vshape = vis.data['vis'].shape
    shape, reffrequency, cellsize, w, imagecentre = create_wcs_from_visibility(vis, params=params)

    vis.data['vis'] = numpy.zeros(vshape)

    spectral_mode = get_parameter(params, 'spectral_mode', 'channel')
    log.debug('fourier_transforms.predict_visibility: spectral mode is %s' % spectral_mode)

    if len(sm.images):
        log.debug("fourier_transforms.predict_visibility: Predicting Visibility from sky model images")

        for im in sm.images:
            wimage = im.wcs

            vshape = vis.data['vis'].shape

            ishape = sm.images[0].data.shape
            log.debug(ishape, vshape)
            nchan = ishape[0]
            npol = ishape[1]
            npixel = ishape[3]
            vshape = vis.data['vis'].shape

            assert ishape[0] == vshape[1], "Image and visibility have different number of polarisations: %d %d" % (
                ishape[1], vshape[2])
            assert ishape[0] == len(vis.frequency), "Image and visibility have different number of channels %d %d" % \
                                                    (ishape[0], len(vis.frequency))
            cellsize = abs(wimage.wcs.cdelt[0]) * numpy.pi / 180.0
            theta = npixel * cellsize
            log.debug("fourier_transforms.predict_visibility: Image cellsize %f radians" % cellsize)
            log.debug("fourier_transforms.predict_visibility: Field of view %f radians" % theta)
            assert (theta / numpy.sqrt(2) < 1.0), "Field of view larger than celestial sphere"

            wstep = get_parameter(params, "wstep", 10000.0)
            wcachesize = int(numpy.ceil(numpy.abs(vis.data['uvw'][:, 2]).max() * reffrequency.value / const.c.value /
                                        wstep))
            log.debug("fourier_transforms.predict_visibility: Making w-kernel cache of %d kernels" % wcachesize)
            wcache = pylru.FunctionCacheManager(lambda iw: wkernaf(N=256, theta=theta, w=iw * wstep, s=15, Qpx=4),
                                                10000)
            predfn = lambda *x: wcachefwd(*x, wstep=wstep, wcache=wcache)

            spectral_mode = get_parameter(params, 'spectral_mode', 'channel')
            log.debug('fourier_transforms.predict_visibility: spectral mode is %s' % spectral_mode)

            if spectral_mode == 'channel':
                for channel in range(nchan):
                    uvw = vis.data['uvw'] * (vis.frequency[channel] / const.c).value
                    for pol in range(npol):
                        log.debug('fourier_transforms.predict_visibility: Predicting from image channel %d, polarisation %d' % (
                        channel, pol))
                        puvw, dv = dopredict(theta, 1.0 / cellsize, uvw, sm.images[0].data[channel, pol, :, :],
                                             predfn=predfn)
                        vis.data['vis'][:, channel, pol] = vis.data['vis'][:, channel, pol] + dv
            else:
                raise NotImplementedError("mode %s not supported" % spectral_mode)

                log.debug("fourier_transforms.predict_visibility: Finished predicting Visibility from sky model images")

    if len(sm.components):
        log.debug("fourier_transforms.predict_visibility: Predicting Visibility from sky model components")

        for icomp in range(len(sm.components)):
            comp = sm.components[icomp]
            cshape = comp.flux.shape
            assert len(cshape) == 2, "Flux should be two dimensional (pol, freq)"
            nchan = cshape[0]
            npol  = cshape[1]

            vshape = vis.data['vis'].shape
            log.debug(vis.data['vis'].shape)

            assert vshape[1] == nchan, "Component %d and visibility %d have different number of channels" % \
                                                    (cshape[0], len(vis.frequency))
            assert vshape[2] == npol, "Component %d and visibility %d have different number of polarisations" % (
                npol, vshape[2])
            # dc = comp.direction.represent_as(CartesianRepresentation)
            l,m,n = skycoord_to_lmn(comp.direction, vis.phasecentre)
            log.debug('fourier_transforms.predict_visibility: Cartesian representation of component %d = (%f, %f, %f)'
                  % (icomp, l,m,n))
            if spectral_mode =='channel':
                for channel in range(nchan):
                    uvw = vis.uvw_lambda(channel)
                    phasor = simulate_point(uvw, l, m)
                    for pol in range(npol):
                        log.debug(
                            'fourier_transforms.predict_visibility: Predicting from component %d channel %d, polarisation %d' % (
                            icomp, channel,
                            pol))
                        vis.vis[:, channel, pol] += comp.flux[channel, pol] * phasor
            else:
                raise NotImplementedError("mode %s not supported" % spectral_mode)

        log.debug("fourier_transforms.predict_visibility: Finished predicting Visibility from sky model components")

    return vis


def weight_visibility(vis, im, params={}):
    """ Reweight the visibility data in place a selected algorithm

    :param vis:
    :type Visibility: Visibility to be processed
    :param im:
    :type Image:
    :param params: Dictionary containing parameters
    :returns: Configuration
    """
    log.error("visibility_operations.weight_visibility: not yet implemented")
    return vis

