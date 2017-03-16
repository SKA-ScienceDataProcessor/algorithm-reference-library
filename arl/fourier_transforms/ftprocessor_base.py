# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing. These are built on top of the core
functions in arl.fourier_transforms.
"""

import collections
from astropy import units as units
from astropy import constants
from astropy import wcs
from astropy.wcs.utils import pixel_to_skycoord

from arl.data.data_models import *
from arl.data.polarisation import convert_pol_frame
from arl.fourier_transforms.convolutional_gridding import fixed_kernel_grid, \
    fixed_kernel_degrid, weight_gridding, w_beam
from arl.fourier_transforms.fft_support import fft, ifft, pad_mid, extract_mid
from arl.fourier_transforms.ftprocessor_params import get_frequency_map, \
    get_polarisation_map, get_uvw_map, get_kernel_list
from arl.image.iterators import *
from arl.image.operations import copy_image, create_empty_image_like
from arl.util.coordinate_support import simulate_point, skycoord_to_lmn
from arl.visibility.coalesce import coalesce_visibility, decoalesce_visibility
from arl.visibility.operations import phaserotate_visibility, copy_visibility
from arl.data.parameters import get_parameter


log = logging.getLogger(__name__)


def shift_vis_to_image(vis, im, tangent=True, inverse=False):
    """Shift visibility to the FFT phase centre of the image in place

    :param vis: Visibility data
    :param im: Image model used to determine phase centre
    :param tangent: Is the shift purely on the tangent plane True|False
    :param inverse: Do the inverse operation True|False
    :returns: visibility with phase shift applied and phasecentre updated

    """
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis

    nchan, npol, ny, nx = im.data.shape
    
    # Convert the FFT definition of the phase center to world coordinates (1 relative)
    # This is the only place the relationship between the image and visibility
    # frames is defined.
    
    image_phasecentre = pixel_to_skycoord(nx // 2, ny // 2, im.wcs, origin=1)
    
    if vis.phasecentre.separation(image_phasecentre).rad > 1e-15:
        if inverse:
            log.debug("shift_vis_from_image: shifting phasecentre from image phase centre %s to visibility phasecentre "
                      "%s" % (image_phasecentre, vis.phasecentre))
        else:
            log.debug("shift_vis_from_image: shifting phasecentre from vis phasecentre %s to image phasecentre %s" %
                      (vis.phasecentre, image_phasecentre))
        vis = phaserotate_visibility(vis, image_phasecentre, tangent=tangent, inverse=inverse)
        vis.phasecentre = im.phasecentre
    
    assert type(vis) is Visibility, "after phase_rotation, vis is not a Visibility"
    
    return vis


def normalize_sumwt(im: Image, sumwt):
    """Normalize out the sum of weights

    :param im: Image, im.data has shape [nchan, npol, ny, nx]
    :param sumwt: Sum of weights [nchan, npol]
    """
    nchan, npol, _, _ = im.data.shape
    assert nchan == sumwt.shape[0]
    assert npol == sumwt.shape[1]
    for chan in range(nchan):
        for pol in range(npol):
            im.data[chan, pol, :, :] /= sumwt[chan, pol]
    return im

def predict_2d_base(vis, model, **kwargs):
    """ Predict using convolutional degridding.

    This is at the bottom of the layering i.e. all transforms are eventually expressed in terms of
    this function. Any shifting needed is performed here.

    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis

    _, _, ny, nx = model.data.shape

    spectral_mode, vfrequencymap = get_frequency_map(vis, model)
    polarisation_mode, vpolarisationmap = get_polarisation_map(vis, model, **kwargs)
    uvw_mode, shape, padding, vuvwmap = get_uvw_map(vis, model, **kwargs)
    kernel_name, gcf, vkernellist = get_kernel_list(vis, model, **kwargs)

    uvgrid = fft((pad_mid(model.data, padding * nx) * gcf).astype(dtype=complex))
    
    vis.data['vis'] = fixed_kernel_degrid(vkernellist, vis.data['vis'].shape, uvgrid,
                                           vuvwmap, vfrequencymap, vpolarisationmap)
    
    # Now we can shift the visibility from the image frame to the original visibility frame
    svis = shift_vis_to_image(vis, model, tangent=True, inverse=True)

    return svis


def predict_2d(vis, model, **kwargs):
    """ Predict using convolutional degridding and w projection
    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_2d: predict using 2d transform")
    return predict_2d_base(vis, model, **kwargs)


def predict_wprojection(vis, model, **kwargs):
    """ Predict using convolutional degridding and w projection
    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_wprojection: predict using wprojection")
    return predict_2d_base(vis, model, kernel='wprojection', **kwargs)


def invert_2d_base(vis, im, dopsf=False, **kwargs):
    """ Invert using 2D convolution function, including w projection optionally

    Use the image im as a template. Do PSF in a separate call.

    This is at the bottom of the layering i.e. all transforms are eventually expressed in terms
    of this function. . Any shifting needed is performed here.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :returns: resulting image

    """
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis

    svis = copy_visibility(vis)
    
    # Shift
    svis = shift_vis_to_image(svis, im, tangent=True, inverse=False)
    
    nchan, npol, ny, nx = im.data.shape
    
    spectral_mode, vfrequencymap = get_frequency_map(vis, im)
    polarisation_mode, vpolarisationmap = get_polarisation_map(vis, im, **kwargs)
    uvw_mode, shape, padding, vuvwmap = get_uvw_map(vis, im, **kwargs)
    kernel_name, gcf, vkernellist = get_kernel_list(vis, im, **kwargs)

   
    # Optionally pad to control aliasing
    imgridpad = numpy.zeros([nchan, npol, padding * ny, padding * nx], dtype='complex')
    if dopsf:
        lvis = numpy.ones_like(svis.data['vis'])
    else:
        lvis = svis.vis
        
    imgridpad, sumwt = fixed_kernel_grid(vkernellist, imgridpad, lvis, svis.data['imaging_weight'], vuvwmap,
                                         vfrequencymap, vpolarisationmap)
    
    # Fourier transform the padded grid to image, multiply by the gridding correction
    # function, and extract the unpadded inner part.

    # Normalise weights for consistency with transform
    sumwt /= float(padding * padding * nx * ny)

    imaginary = get_parameter(kwargs, "imaginary", False)
    if imaginary:
        log.debug("invert_2d_base: retaining imaginary part of dirty image")
        result = extract_mid(ifft(imgridpad) * gcf, npixel=nx)
        return create_image_from_array(result.real, im.wcs), sumwt, create_image_from_array(result.imag, im.wcs)
    else:
        result = extract_mid(numpy.real(ifft(imgridpad)) * gcf, npixel=nx)
        return create_image_from_array(result, im.wcs), sumwt



def invert_2d(vis, im, dopsf=False, **kwargs):
    """ Invert using prolate spheroidal gridding function

    Use the image im as a template. Do PSF in a separate call.

    Note that the image is not normalised but the sum of the weights. This is for ease of use in partitioning.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]

    """
    log.debug("invert_2d: inverting using 2d transform")
    kwargs['kernel'] = get_parameter(kwargs, "kernel", '2d')
    return invert_2d_base(vis, im, dopsf, **kwargs)


def invert_wprojection(vis, im, dopsf=False, **kwargs):
    """ Predict using 2D convolution function, including w projection

    Use the image im as a template. Do PSF in a separate call.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]

    """
    log.debug("invert_2d: inverting using wprojection")
    kwargs['kernel'] = "wprojection"
    return invert_2d_base(vis, im, dopsf, **kwargs)


def predict_skycomponent_blockvisibility(vis: BlockVisibility, sc: Skycomponent, **kwargs) -> BlockVisibility:
    """Predict the visibility from a Skycomponent, add to existing visibility, for BlockVisibility

    :param vis: BlockVisibility
    :param sc: Skycomponent or list of SkyComponents
    :param spectral_mode: {mfs|channel} (channel)
    :returns: BlockVisibility
    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis
    
    if not isinstance(sc, collections.Iterable):
        sc = [sc]
    
    nchan = vis.nchan
    npol = vis.npol

    if not isinstance(sc, collections.Iterable):
        sc = [sc]
    
    k = vis.frequency / constants.c.to('m/s').value
    
    for comp in sc:
    
        assert_same_chan_pol(vis, comp)
        
        flux = comp.flux
        if comp.polarisation_frame != vis.polarisation_frame:
            flux = convert_pol_frame(flux, comp.polarisation_frame, vis.polarisation_frame)

        l, m, n = skycoord_to_lmn(comp.direction, vis.phasecentre)
        for chan in range(nchan):
            phasor = simulate_point(vis.uvw * k, l, m)
            for pol in range(npol):
                vis.data['vis'][...,chan,pol] += flux[chan, pol] * phasor[...]
    
    return vis


def predict_skycomponent_visibility(vis: Visibility, sc: Skycomponent, **kwargs) -> Visibility:
    """Predict the visibility from a Skycomponent, add to existing visibility, for Visibility

    :param vis: Visibility
    :param sc: Skycomponent or list of SkyComponents
    :param spectral_mode: {mfs|channel} (channel)
    :returns: Visibility
    """
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis
    
    if not isinstance(sc, collections.Iterable):
        sc = [sc]
        
    npol = vis.polarisation_frame.npol
    for comp in sc:
        
        l, m, n = skycoord_to_lmn(comp.direction, vis.phasecentre)
        phasor = simulate_point(vis.uvw, l, m)
        
        _, ichan = get_frequency_map(vis, None)
        
        coords = phasor, list(ichan)
        for pol in range(npol):
            vis.data['vis'][:,pol] += [comp.flux[ic, pol] * p for p, ic in zip(*coords)]
    
    return vis


def weight_visibility(vis, im, **kwargs):
    """ Reweight the visibility data using a selected algorithm

    Imaging uses the column "imaging_weight" when imaging. This function sets that column using a
    variety of algorithms

    :param vis:
    :param im:
    :returns: visibility with imaging_weights column added and filled
    """
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis

    spectral_mode, vfrequencymap = get_frequency_map(vis, im)
    polarisation_mode, vpolarisationmap = get_polarisation_map(vis, im, **kwargs)
    uvw_mode, shape, padding, vuvwmap = get_uvw_map(vis, im, **kwargs)

    
    # uvw is in metres, v.frequency / c.value converts to wavelengths, the cellsize converts to phase
    density = None
    densitygrid = None
    
    weighting = get_parameter(kwargs, "weighting", "uniform")
    vis.data['imaging_weight'], density, densitygrid = weight_gridding(im.data.shape, vis.data['weight'], vuvwmap,
                                                                       vfrequencymap, vpolarisationmap, weighting)
    
    return vis, density, densitygrid

def create_image_from_visibility(vis: Visibility, **kwargs) -> Image:
    """Make an from params and Visibility

    :param vis:
    :param phasecentre: Phasecentre (Skycoord)
    :param channel_bandwidth: Channel width (Hz)
    :param cellsize: Cellsize (radians)
    :param npixel: Number of pixels on each axis (512)
    :param frame: Coordinate frame for WCS (ICRS)
    :param equinox: Equinox for WCS (2000.0)
    :param nchan: Number of image channels (Default is 1 -> MFS)
    :returns: image
    """
    assert type(vis) is Visibility or type(vis) is BlockVisibility, \
        "vis is not a Visibility or a BlockVisibility: %r" % (vis)

    log.info("create_image_from_visibility: Parsing parameters to get definition of WCS")
    
    imagecentre = get_parameter(kwargs, "imagecentre", vis.phasecentre)
    phasecentre = get_parameter(kwargs, "phasecentre", vis.phasecentre)
    
    # Spectral processing options
    ufrequency = numpy.unique(vis.frequency)
    vnchan = len(ufrequency)

    frequency = get_parameter(kwargs, "frequency", vis.frequency)
    inchan = get_parameter(kwargs, "nchan", vnchan)
    reffrequency = numpy.min(frequency) * units.Hz
    channel_bandwidth = get_parameter(kwargs, "channel_bandwidth", vis.channel_bandwidth[0]) * units.Hz

    if (inchan == vnchan) and vnchan > 1:
        log.info("create_image_from_visibility: Defining %d channel Image at %s, starting frequency %s, and bandwidth %s"
                 % (inchan, imagecentre, reffrequency, channel_bandwidth))
    elif (inchan == 1) and vnchan > 1:
        assert numpy.abs(channel_bandwidth.value) > 0.0, "Channel width must be non-zero for mfs mode"
        log.info("create_image_from_visibility: Defining single channel MFS Image at %s, starting frequency %s, "
                 "and bandwidth %s"
                 % (imagecentre, reffrequency, channel_bandwidth))
    elif inchan > 1 and vnchan > 1:
        assert numpy.abs(channel_bandwidth.value) > 0.0, "Channel width must be non-zero for mfs mode"
        log.info("create_image_from_visibility: Defining multi-channel MFS Image at %s, starting frequency %s, "
                 "and bandwidth %s"
                 % (imagecentre, reffrequency, channel_bandwidth))
    elif (inchan == 1) and (vnchan == 1):
        assert numpy.abs(channel_bandwidth.value) > 0.0, "Channel width must be non-zero for mfs mode"
        log.info("create_image_from_visibility: Defining single channel Image at %s, starting frequency %s, "
                 "and bandwidth %s"
                 % (imagecentre, reffrequency, channel_bandwidth))
    else:
        raise RuntimeError("create_image_from_visibility: unknown spectral mode ")
    
    # Image sampling options
    npixel = get_parameter(kwargs, "npixel", 512)
    uvmax = numpy.max((numpy.abs(vis.data['uvw'][:, 0:1])))
    log.info("create_image_from_visibility: uvmax = %f wavelengths" % uvmax)
    criticalcellsize = 1.0 / (uvmax * 2.0)
    log.info("create_image_from_visibility: Critical cellsize = %f radians, %f degrees" % (
        criticalcellsize, criticalcellsize * 180.0 / numpy.pi))
    cellsize = get_parameter(kwargs, "cellsize", 0.5 * criticalcellsize)
    log.info("create_image_from_visibility: Cellsize          = %f radians, %f degrees" % (cellsize,
                                                                                           cellsize * 180.0 / numpy.pi))
    if cellsize > criticalcellsize:
        log.info("create_image_from_visibility: Resetting cellsize %f radians to criticalcellsize %f radians" % (
            cellsize, criticalcellsize))
        cellsize = criticalcellsize
    pol_frame = get_parameter(kwargs, "polarisation_frame", PolarisationFrame("stokesI"))
    inpol = pol_frame.npol
    
    # Now we can define the WCS, which is a convenient place to hold the info above
    # Beware of python indexing order! wcs and the array have opposite ordering
    shape = [inchan, inpol, npixel, npixel]
    w = wcs.WCS(naxis=4)
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, 1.0, channel_bandwidth.to(u.Hz).value]
    # The numpy definition of the phase centre of an FFT is n // 2 (0 - rel) so that's what we use for
    # the reference pixel. We have to use 0 rel everywhere.
    w.wcs.crpix = [npixel // 2, npixel // 2, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 1.0, reffrequency.to(u.Hz).value]
    w.naxis = 4
    
    w.wcs.radesys = get_parameter(kwargs, 'frame', 'ICRS')
    w.wcs.equinox = get_parameter(kwargs, 'equinox', 2000.0)
    
    return create_image_from_array(numpy.zeros(shape), wcs=w)


def create_w_term_like(im, w=None, **kwargs):
    """Create an image with a w term phase term in it

    :param im: template image
    :param w: w value to evaluate (default is median abs)
    :returns: Image
    """
    
    fim =  copy_image(im)
    cellsize = abs(fim.wcs.wcs.cdelt[0]) * numpy.pi / 180.0
    _, _, _, npixel = fim.data.shape
    fim.data = w_beam(npixel, npixel * cellsize, w=w)

    fov = npixel * cellsize
    fresnel = numpy.abs(w) * (0.5 * fov) ** 2
    log.info('create_w_term_image: For w = %.1f, field of view = %.6f, Fresnel number = %.2f' % (w, fov, fresnel))
    
    return fim


def create_w_term_image(vis, w=None, **kwargs):
    """Create an image with a w term phase term in it

    :param vis: Visibility
    :param w: w value to evaluate (default is median abs)
    :returns: Image
    """
    if w is None:
        w = numpy.median(numpy.abs(vis.data['uvw'][:, 2]))
        log.info('create_w_term_image: Creating w term image for median w %f' % w)
    
    im = create_image_from_visibility(vis, **kwargs)
    cellsize = abs(im.wcs.wcs.cdelt[0]) * numpy.pi / 180.0
    _, _, _, npixel = im.data.shape
    im.data = w_beam(npixel, npixel * cellsize, w=w)

    fov = npixel * cellsize
    fresnel = numpy.abs(w) * (0.5 * fov) ** 2
    log.info('create_w_term_image: For w = %.1f, field of view = %.6f, Fresnel number = %.2f' % (w, fov, fresnel))

    return im