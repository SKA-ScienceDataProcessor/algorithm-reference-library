# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing. These are built on top of the core
functions in arl.fourier_transforms.
"""

from astropy import units as units
from astropy import wcs
from astropy.constants import c
from astropy.wcs.utils import pixel_to_skycoord

from arl.data.data_models import *
from arl.fourier_transforms.convolutional_gridding import fixed_kernel_grid, \
    fixed_kernel_degrid, weight_gridding, w_beam, anti_aliasing_calculate, anti_aliasing_box
from arl.fourier_transforms.fft_support import fft, ifft, pad_mid, extract_mid
from arl.fourier_transforms.variable_kernels import variable_kernel_grid, variable_kernel_degrid, \
    box_grid, w_kernel_lambda
from arl.image.iterators import *
from arl.util.coordinate_support import simulate_point, skycoord_to_lmn
from arl.visibility.iterators import *
from arl.visibility.operations import phaserotate_visibility

log = logging.getLogger(__name__)


def shift_vis_to_image(vis, im, tangent=True, inverse=False):
    """Shift visibility to the FFT phase centre of the image in place

    :param vis: Visibility data
    :param model: Image model used to determine phase centre
    :returns: visibility with phase shift applied and phasecentre updated

    """
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


def get_ftprocessor_params(vis, model, **kwargs):
    """ Common interface to params for predict and invert

    :param vis: Visibility data
    :param model: Image model used to determine sampling
    :param padding: Pad images by this factor during processing (2)
    :param kernel: kernel to use {2d|wprojection} (2d)
    :param oversampling: Oversampling factor for convolution function (8)
    :param support: Support of convolution function (width = 2*support+2) (3)
    :returns: nchan, npol, ny, nx, shape, spectral_mode, gcf,kernel_type, kernelname, kernel,padding, oversampling, support, cellsize, fov
    """
    
    # Transform parameters
    padding = get_parameter(kwargs, "padding", 2)
    kernelname = get_parameter(kwargs, "kernel", "2d")
    oversampling = get_parameter(kwargs, "oversampling", 8)
    support = get_parameter(kwargs, "support", 3)
    
    # Model image information
    inchan, inpol, ny, nx = model.data.shape
    shape = (padding * ny, padding * nx)
    
    # Visibility information
    nvis, vnchan, vnpol = vis.data['vis'].shape
    
    # UV sampling information
    cellsize = model.wcs.wcs.cdelt[0:2] * numpy.pi / 180.0
    uvscale = numpy.outer(cellsize, vis.frequency / c.value)
    assert uvscale[0, 0] != 0.0, "Error in uv scaling"
    fov = padding * nx * numpy.abs(cellsize[0])
    log.info("get_ftprocessor_params: effective uv cellsize is %.1f wavelengths" % (1.0 / fov))
    
    # Figure out what type of processing we need to do. This is based on the number of channels in
    # the model and in the visibility
    if inchan == 1 and vnchan >= 1:
        spectral_mode = 'mfs'
        log.debug('get_ftprocessor_params: Multi-frequency synthesis mode')
    elif inchan == vnchan and vnchan > 1:
        spectral_mode = 'channel'
        log.debug('get_ftprocessor_params: Channel synthesis mode')
    else:
        spectral_mode = 'channel'
        log.debug('get_ftprocessor_params: Using default channel synthesis mode')
    
    # This information is encapulated in a mapping of vis channel to image channel
    vmap, _ = get_channel_map(vis, model, spectral_mode)
    
    kernel_type = 'fixed'
    gcf = 1.0
    cache = None
    if kernelname == 'wprojection':
        # wprojection needs a lot of commentary!
        log.info("get_ftprocessor_params: using wprojection kernel")
        wmax = numpy.max(numpy.abs(vis.w)) * numpy.max(vis.frequency) / c
        assert wmax > 0, "Maximum w must be > 0.0"
        kernel_type = 'variable'
        r_f = (fov / 2) ** 2 / numpy.abs(cellsize[0])
        log.info("get_ftprocessor_params: Fresnel number = %f" % (r_f))
        delA = get_parameter(kwargs, 'wloss', 0.02)
        # Following equation is from Cornwell, Humphreys, and Voronkov (2012) (equation 24)
        recommended_wstep = numpy.sqrt(2.0 * delA) / (numpy.pi * fov ** 2)
        log.info("get_ftprocessor_params: Recommended wstep = %f" % (recommended_wstep))
        wstep = get_parameter(kwargs, "wstep", recommended_wstep)
        log.info("get_ftprocessor_params: Using w projection with wstep = %f" % (wstep))
        # Now calculate the maximum support for the w kernel
        npixel_kernel = get_parameter(kwargs, "kernelwidth", (int(round(numpy.sin(0.5 * fov) * nx)) // 2))
        log.info("get_ftprocessor_params: w kernel full width = %d pixels" % (npixel_kernel))
        kernel, cache = w_kernel_lambda(vis, shape, fov, wstep=wstep, npixel_kernel=npixel_kernel,
                                    oversampling=oversampling)
        gcf, _ = anti_aliasing_calculate(shape, oversampling)
    elif kernelname == 'box':
        log.info("get_ftprocessor_params: using box car convolution")
        gcf, kernel = anti_aliasing_box(shape)
    else:
        kernelname = '2d'
        gcf, kernel = anti_aliasing_calculate(shape, oversampling)
    
    return vmap, gcf, kernel_type, kernelname, kernel, padding, oversampling, support, \
           cellsize, fov, uvscale, cache


def get_channel_map(vis, im, spectral_mode='channel'):
    """ Get the functions that map channels between image and visibilities

    """
    vis_to_im = lambda chan: chan
    vnchan = vis.data['vis'].shape[1]
    
    # Currently limited to outputing a single MFS channel image
    if spectral_mode == "mfs":
        vis_to_im = lambda chan: 0
        vnchan = im.shape[1]
    elif spectral_mode == 'channel':
        pass
    else:
        log.error("get_channel_map: unknown spectral mode %s" % spectral_mode)
    
    return vis_to_im, vnchan

def log_cacheinfo(cache):
    """ Log info about cache
    
    """
    if cache is not None:
        log.info("log_cacheinfo: final cache statistics = %s" % str(cache.cache_info()))

def predict_2d_base(vis, model, **kwargs):
    """ Predict using convolutional degridding.

    This is at the bottom of the layering i.e. all transforms are eventually expressed in terms of
    this function. Any shifting needed is performed here.

    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    _, _, ny, nx = model.data.shape
    
    vmap, gcf, kernel_type, kernelname, kernel, padding, oversampling, support, cellsize, fov, \
    uvscale, cache = get_ftprocessor_params(vis, model, **kwargs)
    
    uvgrid = fft((pad_mid(model.data, padding * nx) * gcf).astype(dtype=complex))
    
    if kernel_type == 'variable':
        vis.data['vis'] = variable_kernel_degrid(kernel, vis.data['vis'].shape, uvgrid, vis.uvw, uvscale, vmap)
    else:
        vis.data['vis'] = fixed_kernel_degrid(kernel, vis.data['vis'].shape, uvgrid, vis.uvw, uvscale, vmap)
    
    # Now we can shift the visibility from the image frame to the original visibility frame
    vis = shift_vis_to_image(vis, model, tangent=True, inverse=True)
    
    log_cacheinfo(cache)
    
    return vis


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
    svis = copy.deepcopy(vis)
    
    svis = shift_vis_to_image(svis, im, tangent=True, inverse=False)
    
    nchan, npol, ny, nx = im.data.shape
    
    vmap, gcf, kernel_type, kernelname, kernel, padding, oversampling, support, cellsize, \
    fov, uvscale, cache = get_ftprocessor_params(vis, im, **kwargs)
    
    # uvw is in metres, v.frequency / c.value converts to wavelengths, the cellsize converts to phase
    # Optionally pad to control aliasing
    imgridpad = numpy.zeros([nchan, npol, padding * ny, padding * nx], dtype='complex')
    uvw = svis.data['uvw']
    if kernel_type == 'variable':
        if dopsf:
            weights = numpy.ones_like(svis.data['vis'])
            imgridpad, sumwt = variable_kernel_grid(kernel, imgridpad, uvw, uvscale, weights,
                                                    svis.data['imaging_weight'], vmap)
        else:
            imgridpad, sumwt = variable_kernel_grid(kernel, imgridpad, uvw, uvscale, svis.data['vis'],
                                                    svis.data['imaging_weight'], vmap)
    else:
        if dopsf:
            weights = numpy.ones_like(svis.data['vis'])
            if kernelname == 'box':
                imgridpad, sumwt = box_grid(kernel, imgridpad, uvw, uvscale, weights, svis.data['imaging_weight'])
            else:
                imgridpad, sumwt = fixed_kernel_grid(kernel, imgridpad, uvw, uvscale, weights,
                                                     svis.data['imaging_weight'], vmap)
        else:
            if kernelname == 'box':
                imgridpad, sumwt = box_grid(kernel, imgridpad, uvw, uvscale, svis.data['vis'],
                                            svis.data['imaging_weight'])
            else:
                imgridpad, sumwt = fixed_kernel_grid(kernel, imgridpad, uvw, uvscale, svis.data['vis'],
                                                     svis.data['imaging_weight'], vmap)
    
    imgrid = extract_mid(numpy.real(ifft(imgridpad)) * gcf, npixel=nx)
    
    # Normalise weights for consistency with transform
    sumwt /= float(padding * padding * nx * ny)

    log_cacheinfo(cache)

    return create_image_from_array(imgrid, im.wcs), sumwt


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
    result = copy.deepcopy(vis)
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
    assert_same_chan_pol(vis, sc)
    
    l, m, n = skycoord_to_lmn(sc.direction, vis.phasecentre)
    # The data column has vis:[row,nchan,npol], uvw:[row,3]
    for channel in range(sc.nchan):
        uvw = vis.uvw_lambda(channel)
        phasor = simulate_point(uvw, l, m)
        for pol in range(sc.npol):
            vis.vis[:, channel, pol] += sc.flux[channel, pol] * phasor
    
    return vis


def weight_visibility(vis, im, **kwargs):
    """ Reweight the visibility data using a selected algorithm

    Imaging uses the column "imaging_weight" when imaging. This function sets that column using a
    variety of algorithms

    :param vis:
    :param im:
    :returns: visibility with imaging_weights column added and filled
    """
    
    vmap, _, _, _, _, _, _, _, _, _, uvscale, _ = get_ftprocessor_params(vis, im, **kwargs)
    
    # uvw is in metres, v.frequency / c.value converts to wavelengths, the cellsize converts to phase
    density = None
    densitygrid = None
    
    weighting = get_parameter(kwargs, "weighting", "uniform")
    if weighting == 'uniform':
        vis.data['imaging_weight'], density, densitygrid = weight_gridding(im.data.shape, vis.data['uvw'], uvscale,
                                                                           vis.data['weight'], weighting, vmap)
    elif weighting == 'natural':
        vis.data['imaging_weight'] = vis.data['weight']
    else:
        log.error("Unknown visibility weighting algorithm %s" % weighting)
    
    return vis, density, densitygrid


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
    log.info("create_image_from_visibility: Parsing parameters to get definition of WCS")
    
    imagecentre = get_parameter(kwargs, "imagecentre", vis.phasecentre)
    phasecentre = get_parameter(kwargs, "phasecentre", vis.phasecentre)
    
    vnchan = len(vis.frequency)
    inchan = get_parameter(kwargs, "image_nchan", 1)
    reffrequency = numpy.min(vis.frequency) * units.Hz
    deffaultbw = vis.frequency[0]
    if len(vis.frequency) > 1:
        deffaultbw = vis.frequency[1] - vis.frequency[0]
    channelwidth = get_parameter(kwargs, "channelwidth", deffaultbw) * units.Hz
    
    # Spectral processing options
    if (inchan == vnchan) and vnchan > 1:
        log.info("create_image_from_visibility: Defining %d channel Image at %s, starting frequency %s, and bandwidth %s"
                 % (inchan, imagecentre, reffrequency, channelwidth))
    elif (inchan == 1) and vnchan > 1:
        assert numpy.abs(channelwidth.value) > 0.0, "Channel width must be non-zero for mfs mode"
        log.info("create_image_from_visibility: Defining MFS Image at %s, starting frequency %s, and bandwidth %s"
                 % (imagecentre, reffrequency, channelwidth))
    elif (inchan == 1) and (vnchan == 1):
        assert numpy.abs(channelwidth.value) > 0.0, "Channel width must be non-zero for mfs mode"
        log.info("create_image_from_visibility: Defining single channel Image at %s, starting frequency %s, "
                 "and bandwidth %s"
                 % (imagecentre, reffrequency, channelwidth))
    else:
        log.error("create_image_from_visibility: unknown spectral mode ")
    
    # Image sampling options
    npixel = get_parameter(kwargs, "npixel", 512)
    uvmax = (numpy.abs(vis.data['uvw'][:, 0:1]).max() * numpy.max(vis.frequency) / c).value
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
    
    inpol = get_parameter(kwargs, "npol", vis.data['vis'].shape[2])
    
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
    
    fresnel = w * (0.5 * npixel * cellsize) ** 2
    log.info('create_w_term_image: Fresnel number for median w and this field of view and sampling = '
             '%.2f' % (fresnel))
    
    return im
