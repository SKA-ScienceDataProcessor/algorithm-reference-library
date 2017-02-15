# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing. These are built on top of the core
functions in arl.fourier_transforms.
"""

from astropy import units as units
from astropy import wcs
from astropy.wcs.utils import pixel_to_skycoord

from arl.data.data_models import *
from arl.fourier_transforms.convolutional_gridding import fixed_kernel_grid, \
    fixed_kernel_degrid, weight_gridding, w_beam
from arl.fourier_transforms.fft_support import fft, ifft, pad_mid, extract_mid
from arl.fourier_transforms.ftprocessor_params import get_frequency_map, \
    get_polarisation_map, get_uvw_map, get_kernel_list
from arl.image.iterators import *
from arl.image.operations import copy_image
from arl.util.coordinate_support import simulate_point, skycoord_to_lmn
from arl.visibility.coalesce import coalesce_visibility, decoalesce_visibility
from arl.visibility.iterators import *
from arl.visibility.operations import phaserotate_visibility, copy_visibility

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
    # Convert the FFT definition of the phase center to world coordinates (0 relative)
    image_phasecentre = pixel_to_skycoord(ny // 2, nx // 2, im.wcs)
    
    if vis.phasecentre.separation(image_phasecentre).value > 1e-15:
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

    spectral_mode, vfrequencymap = get_frequency_map(vis, model, **kwargs)
    polarisation_mode, vpolarisationmap = get_polarisation_map(vis, model, **kwargs)
    uvw_mode, shape, padding, vuvwmap = get_uvw_map(vis, model, **kwargs)
    kernel_name, gcf, vkernellist = get_kernel_list(vis, model, **kwargs)

    cvis, cindex = coalesce_visibility(vis)

    uvgrid = fft((pad_mid(model.data, padding * nx) * gcf).astype(dtype=complex))
    
    cvis.data['vis'] = fixed_kernel_degrid(vkernellist, cvis.data['vis'].shape, uvgrid, vuvwmap, vfrequencymap,
                                           vpolarisationmap)
    
    dvis = decoalesce_visibility(cvis, vis, cindex=cindex)

    # Now we can shift the visibility from the image frame to the original visibility frame
    svis = shift_vis_to_image(dvis, model, tangent=True, inverse=True)

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
    """ Invert using 2D convolution function, including w projection

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
    
    # Shift and then coalesce to cut down on gridding costs
    shvis = shift_vis_to_image(svis, im, tangent=True, inverse=False)
    svis, _ = coalesce_visibility(shvis, dopsf=dopsf)
    
    nchan, npol, ny, nx = im.data.shape
    
    spectral_mode, vfrequencymap = get_frequency_map(vis, im, **kwargs)
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


def invert_by_image_partitions(vis, im, image_iterator=raster_iter, dopsf=False,
                               invert_function=invert_2d, **kwargs):
    """ Predict using image partitions, calling specified predict function

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param image_iterator: Iterator to use for partitioning
    :param dopsf: Make the psf instead of the dirty image
    :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]
    """
    
    log.debug("invert_by_image_partitions: Inverting by image partitions")
    i = 0
    nchan, npol, _, _ = im.shape
    totalwt = numpy.zeros([nchan, npol])
    for dpatch in image_iterator(im, **kwargs):
        result, sumwt = invert_function(vis, dpatch, dopsf, **kwargs)
        totalwt = sumwt
        # Ensure that we fill in the elements of dpatch instead of creating a new numpy arrray
        dpatch.data[...] = result.data[...]
        assert numpy.max(numpy.abs(dpatch.data)), "Partition image %d appears to be empty" % i
        i += 1
    assert numpy.max(numpy.abs(im.data)), "Output image appears to be empty"
    
    # Loose thread here: we have to assume that all patchs have the same sumwt
    return im, totalwt


def invert_by_vis_partitions(vis, im, vis_iterator, dopsf=False, invert_function=invert_2d, **kwargs):
    """ Invert using wslices

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :returns: resulting image

    """
    log.debug("invert_by_vis_partitions: Inverting by vis partitions")
    nchan, npol, _, _ = im.shape
    result = None
    totalwt = numpy.zeros([nchan, npol])
    
    for visslice in vis_iterator(vis, **kwargs):
        result, sumwt = invert_function(visslice, im, dopsf, invert_function, **kwargs)
        totalwt += sumwt
    
    return result, totalwt


def predict_by_image_partitions(vis, model, image_iterator=raster_iter, predict_function=predict_2d,
                                **kwargs):
    """ Predict using image partitions, calling specified predict function

    :param vis: Visibility to be predicted
    :param model: model image
    :param image_iterator: Image iterator used to access the image
    :param predict_function: Function to be used for prediction (allows nesting)
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_by_image_partitions: Predicting by image partitions")
    vis.data['vis'] *= 0.0
    result = copy_visibility(vis)
    for dpatch in image_iterator(model, **kwargs):
        result = predict_function(result, dpatch, **kwargs)
        vis.data['vis'] += result.data['vis']
    return vis


def predict_by_vis_partitions(vis, model, vis_iterator, predict_function=predict_2d, **kwargs):
    """ Predict using vis partitions

    :param vis: Visibility to be predicted
    :param model: model image
    :param vis_iterator: Iterator to use for partitioning
    :param predict_function: Function to be used for prediction (allows nesting)
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_by_vis_partitions: Predicting by vis partitions")
    for vslice in vis_iterator(vis, **kwargs):
        vis.data['vis'] += predict_function(vslice, model, **kwargs).data['vis']
    return vis


def predict_skycomponent_visibility(vis: Visibility, sc: Skycomponent, **kwargs) -> Visibility:
    """Predict the visibility from a Skycomponent, add to existing visibility

    :param vis: Visibility
    :param sc: Skycomponent
    :param spectral_mode: {mfs|channel} (channel)
    :returns: Visibility
    """
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis

    l, m, n = skycoord_to_lmn(sc.direction, vis.phasecentre)
    phasor = simulate_point(vis.uvw, l, m)

    _, ipol = get_polarisation_map(vis)
    _, ichan = get_frequency_map(vis)
    
    coords = phasor, list(ichan), list(ipol)
    vis.data['vis'] += [sc.flux[ic,ip] * p for p, ic, ip in zip(*coords)]
    
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

    spectral_mode, vfrequencymap = get_frequency_map(vis, im, **kwargs)
    polarisation_mode, vpolarisationmap = get_polarisation_map(vis, im, **kwargs)
    uvw_mode, shape, padding, vuvwmap = get_uvw_map(vis, im, **kwargs)

    
    # uvw is in metres, v.frequency / c.value converts to wavelengths, the cellsize converts to phase
    density = None
    densitygrid = None
    
    weighting = get_parameter(kwargs, "weighting", "uniform")
    if weighting == 'uniform':
        vis.data['imaging_weight'], density, densitygrid = weight_gridding(im.data.shape,
                                                                           vis.data['weight'],
                                                                           vuvwmap, vfrequencymap, vpolarisationmap,
                                                                           weighting)
    elif weighting == 'natural':
        vis.data['imaging_weight'] = vis.data['weight']
    else:
        raise RuntimeError("Unknown visibility weighting algorithm %s" % weighting)
    
    return vis, density, densitygrid


# noinspection PyStringFormat,PyStringFormat
def create_image_from_visibility(vis, **kwargs):
    """Make an from params and Visibility

    :param vis:
    :param phasecentre: Phasecentre (Skycoord)
    :param channelwidth: Channel width (Hz)
    :param cellsize: Cellsize (radians)
    :param npixel: Number of pixels on each axis (512)
    :param frame: Coordinate frame for WCS (ICRS)
    :param equinox: Equinox for WCS (2000.0)
    :param image_nchan: Number of image channels (Default is 1 -> MFS)
    :returns: image
    """
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis

    log.info("create_image_from_blockvisibility: Parsing parameters to get definition of WCS")
    
    imagecentre = get_parameter(kwargs, "imagecentre", vis.phasecentre)
    phasecentre = get_parameter(kwargs, "phasecentre", vis.phasecentre)
    
    # Spectral processing options
    vnchan = len(numpy.unique(vis.frequency))
    inchan = get_parameter(kwargs, "image_nchan", 1)
    reffrequency = numpy.min(vis.frequency) * units.Hz
    channelwidth = get_parameter(kwargs, "channelwidth", 1e6) * units.Hz

    if (inchan == vnchan) and vnchan > 1:
        log.info("create_image_from_blockvisibility: Defining %d channel Image at %s, starting frequency %s, and bandwidth %s"
                 % (inchan, imagecentre, reffrequency, channelwidth))
    elif (inchan == 1) and vnchan > 1:
        assert numpy.abs(channelwidth.value) > 0.0, "Channel width must be non-zero for mfs mode"
        log.info("create_image_from_blockvisibility: Defining MFS Image at %s, starting frequency %s, and bandwidth %s"
                 % (imagecentre, reffrequency, channelwidth))
    elif (inchan == 1) and (vnchan == 1):
        assert numpy.abs(channelwidth.value) > 0.0, "Channel width must be non-zero for mfs mode"
        log.info("create_image_from_blockvisibility: Defining single channel Image at %s, starting frequency %s, "
                 "and bandwidth %s"
                 % (imagecentre, reffrequency, channelwidth))
    else:
        raise RuntimeError("create_image_from_blockvisibility: unknown spectral mode ")
    
    # Image sampling options
    npixel = get_parameter(kwargs, "npixel", 512)
    uvmax = numpy.max((numpy.abs(vis.data['uvw'][:, 0:1])))
    log.info("create_image_from_blockvisibility: uvmax = %f wavelengths" % uvmax)
    criticalcellsize = 1.0 / (uvmax * 2.0)
    log.info("create_image_from_blockvisibility: Critical cellsize = %f radians, %f degrees" % (
        criticalcellsize, criticalcellsize * 180.0 / numpy.pi))
    cellsize = get_parameter(kwargs, "cellsize", 0.5 * criticalcellsize)
    log.info("create_image_from_blockvisibility: Cellsize          = %f radians, %f degrees" % (cellsize,
                                                                                           cellsize * 180.0 / numpy.pi))
    if cellsize > criticalcellsize:
        log.info("create_image_from_blockvisibility: Resetting cellsize %f radians to criticalcellsize %f radians" % (
            cellsize[1], criticalcellsize[1]))
        cellsize = criticalcellsize
    
    inpol = get_parameter(kwargs, "npol", 1)
    
    # Now we can define the WCS, which is a convenient place to hold the info above
    # Beware of python indexing order! wcs and the array have opposite ordering
    shape = [inchan, inpol, npixel, npixel]
    w = wcs.WCS(naxis=4)
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, 1.0, channelwidth.value]
    # The numpy definition of the phase centre of an FFT is n // 2 so that's what we use for
    # the reference pixel but we have to use 1-relative indxing for wcs (arghhh!)
    w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.value, phasecentre.dec.value, 1.0, reffrequency.value]
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
