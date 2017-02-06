# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing. These are built on top of the core
functions in arl.fourier_transforms.
"""

from astropy import units as units
from astropy import wcs
from astropy.constants import c

from arl.data.data_models import *
from arl.fourier_transforms.convolutional_gridding import anti_aliasing_calculate, anti_aliasing_box
from arl.fourier_transforms.variable_kernels import w_kernel_lambda
from arl.image.iterators import *

log = logging.getLogger(__name__)


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
        raise RuntimeError("Unknown spectral mode %s" % spectral_mode)
    
    return vis_to_im, vnchan


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

