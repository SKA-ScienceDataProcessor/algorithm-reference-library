# Tim Cornwell <realtimcornwell@gmail.com>
#
# Synthesis imaging functions
#

import logging

from astropy import units as units
from astropy import wcs

from arl.data_models import *
from arl.image_operations import create_empty_image_like
from arl.parameters import *
from arl.coordinate_support import simulate_point, skycoord_to_lmn

log = logging.getLogger("arl.fourier_transforms")

"""
Functions that perform imaging i.e. conversion of an Image to/from a Visibility
"""

def _create_wcs_from_visibility(vis, params={}):
    """Make a world coordinate system from params and Visibility

    :param vis:
    :type Visibility: Visibility to be processed
    :param params: keyword=value parameters
    :returns: WCS
    """
    log_parameters(params)
    log.debug("fourier_transforms.create_wcs_from_visibility: Parsing parameters to get definition of WCS")
    imagecentre = get_parameter(params, "imagecentre", vis.phasecentre)
    phasecentre = get_parameter(params, "phasecentre", vis.phasecentre)
    reffrequency = get_parameter(params, "reffrequency", numpy.max(vis.frequency)) * units.Hz
    deffaultbw = vis.frequency[0]
    if len(vis.frequency) > 1:
        deffaultbw = vis.frequency[1] - vis.frequency[0]
    channelwidth = get_parameter(params, "channelwidth", deffaultbw) * units.Hz
    log.debug("fourier_transforms.create_wcs_from_visibility: Defining Image at %s, frequency %s, and bandwidth %s"
          % (imagecentre, reffrequency, channelwidth))

    npixel = get_parameter(params, "npixel", 512)
    uvmax = (numpy.abs(vis.data['uvw']).max() * reffrequency / const.c).value
    log.debug("create_wcs_from_visibility: uvmax = %f lambda" % uvmax)
    criticalcellsize = 1.0 / (uvmax * 2.0)
    log.debug("create_wcs_from_visibility: Critical cellsize = %f radians, %f degrees" % (
        criticalcellsize, criticalcellsize * 180.0 / numpy.pi))
    cellsize = get_parameter(params, "cellsize", 0.5 * criticalcellsize)
    log.debug("create_wcs_from_visibility: Cellsize          = %f radians, %f degrees" % (cellsize,
                                                                                       cellsize * 180.0 / numpy.pi))
    if cellsize > criticalcellsize:
        log.debug("Resetting cellsize %f radians to criticalcellsize %f radians" % (cellsize, criticalcellsize))
        cellsize = criticalcellsize

    npol = 4
    # Beware of python indexing order! wcs and the array have opposite ordering
    shape = [len(vis.frequency), npol, npixel, npixel]
    w = wcs.WCS(naxis=4)
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, 1.0, channelwidth.value]
    w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.value, phasecentre.dec.value, 1.0, reffrequency.value]
    w.naxis = 4

    w.wcs.radesys = get_parameter(params, 'frame', 'ICRS')
    w.wcs.equinox = get_parameter(params, 'equinox', 2000.0)

    return shape, reffrequency, cellsize, w, imagecentre


def invert_visibility(vis, model, params={}):
    """Invert to make dirty Image and PSF
    
    This is the top level invert routine.

    :param vis:
    :type Visibility: Visibility to be processed
    :param model: Template model
    :returns: (dirty image, psf)
    """
    log_parameters(params)
    log.debug("invert_visibility: Inverting Visibility to make dirty and psf")
    shape, reffrequency, cellsize, wcs, imagecentre = _create_wcs_from_visibility(vis, params=params)

    npixel = shape[3]
    field_of_view = npixel * cellsize

    log.debug("invert_visibility: Specified npixel=%d, cellsize = %f rad, FOV = %f rad" %
          (npixel, cellsize, field_of_view))

    dirty = create_empty_image_like(model)
    d = dirty.data
    psf = create_empty_image_like(model)
    p = psf.data

    spectral_mode = get_parameter(params, 'spectral_mode', 'channel')
    log.debug('invert_visibility: spectral mode is %s' % spectral_mode)

    if spectral_mode == 'channel':
        pmax = 0.0
        nchan = shape[0]
        npol = shape[1]
        # These loops should ideally be at the bottom of the stack
        for channel in range(nchan):
            for pol in range(npol):
                log.debug('invert_visibility: Inverting channel %d, polarisation %d' % (channel, pol))
                # d[channel, pol, :, :], p[channel, 0, :, :], pmax = \
                #     invert2d(field_of_view, 1.0 / cellsize, vis.uvw_lambda(channel),
                #                vis.vis[:, channel, pol],  model, params=params)
            assert pmax > 0.0, ("No data gridded for channel %d" % channel)
    else:
        raise NotImplementedError("mode %s not supported" % spectral_mode)

    log.debug("invert_visibility: Finished making dirty and psf")

    return dirty, psf, pmax

def predict_visibility(vis: Visibility, sm: SkyModel, params={}) -> Visibility:
    """Predict the visibility from a SkyModel including both components and images

    :param vis:
    :type Visibility: Visibility to be processed
    :param sm:
    :type SkyModel:
    :returns: Visibility
    """
    shape, reffrequency, cellsize, w, imagecentre = _create_wcs_from_visibility(vis, params=params)

    vis.data['vis'] = 0.0 * vis.data['vis']

    spectral_mode = get_parameter(params, 'spectral_mode', 'channel')
    log.debug('predict_visibility: spectral mode is %s' % spectral_mode)

    if len(sm.images):
        log.debug("predict_visibility: Predicting Visibility from sky model images")

        for im in sm.images:
            assert_same_chan_pol(vis, im)

            # Determine image size
            cellsize = abs(im.wcs.wcs.cdelt[0]) * numpy.pi / 180.0
            field_of_view = im.npixel * cellsize
            log.debug("predict_visibility: Image cellsize %f radians" % cellsize)
            log.debug("predict_visibility: Field of view %f radians" % field_of_view)
            assert (field_of_view / numpy.sqrt(2) < 1.0), "Field of view larger than celestial sphere"

            spectral_mode = get_parameter(params, 'spectral_mode', 'channel')
            log.debug('predict_visibility: spectral mode is %s' % spectral_mode)

            if spectral_mode == 'channel':
                for channel in range(im.nchan):
                    uvw = vis.uvw_lambda(channel)
                    for pol in range(im.npol):
                        log.debug('predict_visibility: Predicting from image channel %d, polarisation %d' % (
                        channel, pol))
                        img = sm.images[0].data[channel, pol, :, :]
            #             dv = predict_image_partition(field_of_view, 1.0 / cellsize, numpy.array(uvw), img)
            #             vis.vis[:, channel, pol] += dv
            else:
                raise NotImplementedError("mode %s not supported" % spectral_mode)

            log.debug("fourier_transforms.predict_visibility: Finished predicting Visibility from sky model images")

    # Now do the components (point sources only at the moment)
    if len(sm.components):
        log.debug("fourier_transforms.predict_visibility: Predicting Visibility from sky model components")

        for icomp, comp in enumerate(sm.components):

            log.debug("fourier_transforms.predict_visibility: visibility shape = %s" % str(vis.vis.shape))
            assert_same_chan_pol(vis, comp)

            l,m,n = skycoord_to_lmn(comp.direction, vis.phasecentre)
            log.debug('fourier_transforms.predict_visibility: Cartesian representation of component %d = (%f, %f, %f)'
                  % (icomp, l,m,n))

            if spectral_mode =='channel':
                for channel in range(comp.nchan):
                    uvw = vis.uvw_lambda(channel)
                    phasor = simulate_point(uvw, l, m)
                    for pol in range(comp.npol):
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
    # TODO: implement

    log_parameters(params)
    log.error("fourier_transforms.weight_visibility: not yet implemented")
    return vis

