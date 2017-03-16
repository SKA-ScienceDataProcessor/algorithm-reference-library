# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing. These are built on top of the core
functions in arl.fourier_transforms.
"""

from arl.data.data_models import *
from arl.data.parameters import *
from arl.fourier_transforms.convolutional_gridding import anti_aliasing_calculate, w_kernel
from arl.image.iterators import *

log = logging.getLogger(__name__)


def get_frequency_map(vis, im=None, **kwargs):
    """ Map channels from visibilities to image

    """
    
    # Find the unique frequencies in the visibility
    ufrequency = numpy.unique(vis.frequency)
    vnchan = len(ufrequency)

    if im is None:
        spectral_mode = 'channel'
        vfrequencymap = get_rowmap(vis.frequency, ufrequency)
        
    elif im.data.shape[0] == 1 and vnchan >= 1:
        spectral_mode = 'mfs'
        vfrequencymap = numpy.zeros_like(vis.frequency, dtype='int')
    else:
        # We can map these to image channels
        v2im_map = im.wcs.sub(['spectral']).wcs_world2pix(ufrequency, 0)[0].astype('int')
    
        spectral_mode = 'channel'
        nrows = len(vis.frequency)
        row2vis = numpy.array(get_rowmap(vis.frequency, ufrequency))
        vfrequencymap = [v2im_map[row2vis[row]] for row in range(nrows)]
    
    return spectral_mode, vfrequencymap


def get_polarisation_map(vis: Visibility, im: Image=None, **kwargs):
    """ Get the mapping of visibility polarisations to image polarisations
    
    """
    if vis.polarisation_frame == im.polarisation_frame:
        if vis.polarisation_frame == PolarisationFrame('stokesI'):
            return "stokesI->stokesI", lambda pol: 0
        elif vis.polarisation_frame == PolarisationFrame('stokesIQUV'):
            return "stokesIQUV->stokesIQUV", lambda pol: pol

    return "unknown", lambda pol: pol


def get_rowmap(col, ucol=None):
    """ Get a generator to map unique cols
    
    :param col: Data column
    :param ucol: Unique values in col
    """
    pdict = {}
    
    def phash(f):
        return numpy.round(f).astype('int')
    
    if ucol is None:
        ucol = numpy.unique(col)
        
    for i, f in enumerate(ucol):
        pdict[phash(f)] = i
    vmap = [pdict[phash(p)] for p in col]
    return vmap


def get_uvw_map(vis, im, **kwargs):
    """ Get the generators that map channels uvw to pixels

    """
    # Transform parameters
    padding = get_parameter(kwargs, "padding", 2)
    
    # Model image information
    inchan, inpol, ny, nx = im.data.shape
    shape = (1, padding * ny, padding * nx)
    # UV sampling information
    uvwscale = numpy.zeros([3])
    uvwscale[0:2] = im.wcs.wcs.cdelt[0:2] * numpy.pi / 180.0
    assert uvwscale[0] != 0.0, "Error in uv scaling"
    fov = padding * nx * numpy.abs(uvwscale[0])
    
    vuvwmap = uvwscale * vis.uvw
    uvw_mode = "2d"
    
    return uvw_mode, shape, padding, vuvwmap


def standard_kernel_list(vis, shape, oversampling=8, support=3):
    """Return a lambda function to calculate the standard visibility kernel

    :param vis: visibility
    :param shape: tuple with 2D shape of grid
    :param oversampling: Oversampling factor
    :param support: Support of kernel
    :returns: Function to look up gridding kernel
    """
    return [anti_aliasing_calculate(shape, oversampling, support)[1]]


def w_kernel_list(vis, shape, fov, oversampling=4, wstep=100.0, npixel_kernel=16):
    """Return a generator for the w kernel for each row

    This function is called once. It uses an LRU cache to hold the convolution kernels. As a result,
    initially progress is slow as the cache is filled. Then it speeds up.

    :param vis: visibility
    :param shape: tuple with 2D shape of grid
    :param fov: Field of view in radians
    :param oversampling: Oversampling factor
    :param wstep: Step in w between cached functions
    :returns: Function to look up gridding kernel as function of row, and cache
    """
    wmax = numpy.max(numpy.abs(vis.w))
    log.debug("w_kernel_list: Maximum w = %.1f , step is %.1f wavelengths" % (wmax, wstep))
    
    def digitise_w(w):
        return numpy.round(w / wstep).astype('int')
    
    # Use a dictionary but look at performance
    kernels = {}
    wint_list = numpy.unique(digitise_w(vis.w))
    for wint in wint_list:
        kernels[wint] = w_kernel(field_of_view=fov, w=wstep * wint, npixel_farfield=shape[0],
                                 npixel_kernel=npixel_kernel, kernel_oversampling=oversampling)
    # We will return a generator that can be instantiated at the last moment. The memory for
    # the kernels is needed but the pointer per row can be deferred.
    w_kernels = (kernels[digitise_w(w)] for w in vis.w)
    
    return w_kernels


def get_kernel_list(vis, im, **kwargs):
    """Get the list of kernels, one per visibility
    
    """
    
    shape = im.data.shape
    npixel = shape[3]
    cellsize = numpy.pi * im.wcs.wcs.cdelt[1] / 180.0
    
    kernelname = get_parameter(kwargs, "kernel", "2d")
    oversampling = get_parameter(kwargs, "oversampling", 8)
    padding = get_parameter(kwargs, "padding", 2)
    
    gcf, _ = anti_aliasing_calculate((padding * npixel, padding * npixel), oversampling)
    
    if kernelname == 'wprojection':
        
        # wprojection needs a lot of commentary!
        log.info("get_kernel_list: Using wprojection kernel")
        wmax = numpy.max(numpy.abs(vis.w))
        assert wmax > 0, "Maximum w must be > 0.0"
        
        # The field of view must be as padded!
        fov = cellsize * npixel * padding
        r_f = (fov / 2) ** 2 / abs(cellsize)
        log.info("get_kernel_list: Fresnel number = %f" % (r_f))
        delA = get_parameter(kwargs, 'wloss', 0.02)
        
        # Following equation is from Cornwell, Humphreys, and Voronkov (2012) (equation 24)
        recommended_wstep = numpy.sqrt(2.0 * delA) / (numpy.pi * fov ** 2)
        log.info("get_kernel_list: Recommended wstep = %f" % (recommended_wstep))
        wstep = get_parameter(kwargs, "wstep", recommended_wstep)
        log.info("get_kernel_list: Using w projection with wstep = %f" % (wstep))
        
        # Now calculate the maximum support for the w kernel
        npixel_kernel = get_parameter(kwargs, "kernelwidth", (2 * int(round(numpy.sin(0.5 * fov) * npixel/4.0))))
        assert npixel_kernel % 2 == 0
        log.info("get_kernel_list: Maximum w kernel full width = %d pixels" % (npixel_kernel))
        kernel_list = w_kernel_list(vis, (npixel, npixel), fov, wstep=wstep,
                                    npixel_kernel=npixel_kernel, oversampling=oversampling)
    else:
        kernelname = '2d'
        kernel_list = standard_kernel_list(vis, (padding * npixel, padding * npixel), oversampling=8, support=3)
    
    return kernelname, gcf, kernel_list
