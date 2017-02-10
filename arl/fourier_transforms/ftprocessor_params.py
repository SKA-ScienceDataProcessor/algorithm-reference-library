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


def get_frequency_map(vis, im=None, **kwargs):
    """ Get the functions that map channels between image and visibilities

    """

    ufrequency = numpy.unique(vis.frequency)
    vnchan = len(ufrequency)

    if im is not None and im.data.shape[1] ==1 and vnchan >=1:
        spectral_mode = 'mfs'
        vfrequencymap = lambda f: 0
        log.debug('get_ftprocessor_params: Multi-frequency synthesis mode')
    else:
        spectral_mode = 'channel'
        log.debug('get_ftprocessor_params: Channel synthesis mode')
        fdict = {}
        for chan, f in enumerate(ufrequency):
            fdict[numpy.round(f).astype('int')] = chan
        vfrequencymap = lambda frequency: numpy.array([fdict[numpy.round(f).astype('int')] for f in frequency])


    return spectral_mode, vfrequencymap

def get_polarisation_map(vis, im=None, **kwargs):
    """ Get the mapping of visibility polarisations to image polarisations
    
    """
    return "direct", lambda pol: pol


def get_uvw_map(vis, im, **kwargs):
    """ Get the function that map channels uvw to pixels

    """
    # Transform parameters
    padding = get_parameter(kwargs, "padding", 2)
    kernelname = get_parameter(kwargs, "kernel", "2d")
    oversampling = get_parameter(kwargs, "oversampling", 8)
    support = get_parameter(kwargs, "support", 3)

    # Model image information
    inchan, inpol, ny, nx = im.data.shape
    shape = (1, padding * ny, padding * nx)
    # UV sampling information
    uvwscale = numpy.zeros([3])
    uvwscale[0:2] = im.wcs.wcs.cdelt[0:2] * numpy.pi / 180.0
    assert uvwscale[0] != 0.0, "Error in uv scaling"
    fov = padding * nx * numpy.abs(uvwscale[0])
    log.info("get_ftprocessor_params: effective uv cellsize is %.1f wavelengths" % (1.0 / fov))
    
    vuvwmap = lambda uvw: uvwscale * uvw
    uvw_mode = "2d"

    return uvw_mode, shape, vuvwmap


def get_ftprocessor_params(vis, model, **kwargs):
    """ Common interface to params for predict and invert
    
    The main task of this function is to construct a set of mappings between vis and image. For example,
    given a frequency, we need to know which image channel is maps to. Similarly polarisation.

    :param vis: CompressedVisibility data
    :param model: Image model used to determine sampling
    :param padding: Pad images by this factor during processing (2)
    :param kernel: kernel to use {2d|wprojection} (2d)
    :param oversampling: Oversampling factor for convolution function (8)
    :param support: Support of convolution function (width = 2*support+2) (3)
    :returns: nchan, npol, ny, nx, shape, spectral_mode, gcf,kernel_type, kernelname, kernel,padding, oversampling, support, cellsize, fov
    """

    assert type(vis) is CompressedVisibility, "vis is not a CompressedVisibility: %r" % vis

    # Transform parameters
    padding = get_parameter(kwargs, "padding", 2)
    kernelname = get_parameter(kwargs, "kernel", "2d")
    oversampling = get_parameter(kwargs, "oversampling", 8)
    support = get_parameter(kwargs, "support", 3)
    
    # Model image information
    inchan, inpol, ny, nx = model.data.shape
        
    # Figure out what type of processing we need to do. This is based on the number of channels in
    # the model and in the visibility
    
    # This information is encapulated in a mapping of vis channel to image channel
    spectral_mode, vfrequencymap = get_frequency_map(vis, model, **kwargs)
    polarisation_mode, vpolarisationmap = get_polarisation_map(vis, model, **kwargs)
    uvw_mode, shape, vuvwmap = get_uvw_map(vis, model, **kwargs)

    kernel_type = 'fixed'
    gcf = 1.0
    cache = None
    if kernelname == 'wprojection':
        # wprojection needs a lot of commentary!
        log.info("get_ftprocessor_params: using wprojection kernel")
        wmax = numpy.max(numpy.abs(vis.w))
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
    
    return vfrequencymap, vpolarisationmap, vuvwmap, gcf, kernel_type, kernelname, kernel, padding, oversampling, \
           support, cellsize, fov, uvscale, cache

