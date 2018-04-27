"""
Functions that aid definition of fourier transform processing.
"""

import logging
import warnings

from astropy.wcs import FITSFixedWarning

import numpy

from data_models.memory_data_models import Visibility, Image
from data_models.parameters import get_parameter
from data_models.polarisation import PolarisationFrame

from ..fourier_transforms.convolutional_gridding import anti_aliasing_calculate
from ..image.operations import convert_image_to_kernel
from ..image.operations import copy_image, fft_image, pad_image, create_w_term_like

log = logging.getLogger(__name__)


def get_frequency_map(vis, im: Image = None):
    """ Map channels from visibilities to image

    """
    
    # Find the unique frequencies in the visibility
    ufrequency = numpy.unique(vis.frequency)
    vnchan = len(ufrequency)
    
    if im is None:
        spectral_mode = 'channel'
        vfrequencymap = get_rowmap(vis.frequency, ufrequency)
        assert min(vfrequencymap) >= 0, "Invalid frequency map: visibility channel < 0: %s" % str(vfrequencymap)
    
    elif im.data.shape[0] == 1 and vnchan >= 1:
        spectral_mode = 'mfs'
        vfrequencymap = numpy.zeros_like(vis.frequency, dtype='int')
    
    else:
        # We can map these to image channels
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FITSFixedWarning)
            v2im_map = im.wcs.sub(['spectral']).wcs_world2pix(ufrequency, 0)[0].astype('int')
        
        spectral_mode = 'channel'
        nrows = len(vis.frequency)
        row2vis = numpy.array(get_rowmap(vis.frequency, ufrequency))
        vfrequencymap = [v2im_map[row2vis[row]] for row in range(nrows)]
        
        assert min(vfrequencymap) >= 0, "Invalid frequency map: image channel < 0 %s" % str(vfrequencymap)
        assert max(vfrequencymap) < im.shape[0], "Invalid frequency map: image channel > number image channels %s" % \
                                                 str(vfrequencymap)
    
    return spectral_mode, vfrequencymap


def get_polarisation_map(vis: Visibility, im: Image = None):
    """ Get the mapping of visibility polarisations to image polarisations
    
    """
    if vis.polarisation_frame == im.polarisation_frame:
        if vis.polarisation_frame == PolarisationFrame('stokesI'):
            return "stokesI->stokesI", lambda pol: 0
        elif vis.polarisation_frame == PolarisationFrame('stokesIQUV'):
            return "stokesIQUV->stokesIQUV", lambda pol: pol
    
    return "unknown", lambda pol: pol


def get_rowmap(col, ucol=None):
    """ Map to unique cols
    
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
    vmap = []
    for p in col:
        vmap.append(pdict[phash(p)])

    return vmap


def get_uvw_map(vis: Visibility, im: Image, padding=2):
    """ Get the generators that map channels uvw to pixels

    :param padding:
    :return: uvw mode, shape, padding, uvw mapping
    """
    # Transform parameters
    
    # Model image information
    inchan, inpol, ny, nx = im.data.shape
    shape = (1, int(round(padding * ny)), int(round(padding * nx)))
    # UV sampling information
    uvwscale = numpy.zeros([3])
    uvwscale[0:2] = im.wcs.wcs.cdelt[0:2] * numpy.pi / 180.0
    assert uvwscale[0] != 0.0, "Error in uv scaling"
    
    vuvwmap = uvwscale * vis.uvw
    uvw_mode = "2d"
    
    return uvw_mode, shape, padding, vuvwmap


def standard_kernel_list(vis: Visibility, shape, oversampling=8, support=3):
    """Return a generator to calculate the standard visibility kernel

    :param vis: visibility
    :param shape: tuple with 2D shape of grid
    :param oversampling: Oversampling factor
    :param support: Support of kernel
    :return: Function to look up gridding kernel
    """
    return numpy.zeros_like(vis.w, dtype='int'), [anti_aliasing_calculate(shape, oversampling, support)[1]]


def w_kernel_list(vis: Visibility, im: Image, oversampling=1, wstep=50.0, kernelwidth=16, **kwargs):
    """ Calculate w convolution kernels
    
    Uses create_w_term_like to calculate the w screen. This is exactly as wstacking does.

    Returns (indices to the w kernel for each row, kernels)

    Each kernel has axes [centre_v, centre_u, offset_v, offset_u]. We currently use the same
    convolution function for all channels and polarisations. Changing that behaviour would
    require modest changes here and to the gridding/degridding routines.

    :param vis: visibility
    :param image: Template image (padding, if any, occurs before this)
    :param oversampling: Oversampling factor
    :param wstep: Step in w between cached functions
    :return: (indices to the w kernel for each row, kernels)
    """

    nchan, npol, ny, nx = im.shape
    gcf, _ = anti_aliasing_calculate((ny, nx))

    assert oversampling % 2 == 0 or oversampling == 1, "oversampling must be unity or even"
    assert kernelwidth % 2 == 0, "kernelwidth must be even"

    wmaxabs = numpy.max(numpy.abs(vis.w))
    log.debug("w_kernel_list: Maximum absolute w = %.1f, step is %.1f wavelengths" % (wmaxabs, wstep))

    def digitise(w, wstep):
        return numpy.ceil((w + wmaxabs) / wstep).astype('int')
    
    # Find all the unique indices for which we need a kernel
    nwsteps = digitise(wmaxabs, wstep) + 1
    w_list = numpy.linspace(-wmaxabs, +wmaxabs, nwsteps)
    
    wtemplate = copy_image(im)
    
    wtemplate.data = numpy.zeros(wtemplate.shape, dtype=im.data.dtype)
    
    padded_shape = list(wtemplate.shape)
    padded_shape[3] *= oversampling
    padded_shape[2] *= oversampling

    # For all the unique indices, calculate the corresponding w kernel
    kernels = list()
    for w in w_list:
        # Make a w screen
        wscreen = create_w_term_like(wtemplate, w, vis.phasecentre, **kwargs)
        wscreen.data /= gcf
        assert numpy.max(numpy.abs(wscreen.data)) > 0.0, 'w screen is empty'
        wscreen_padded = pad_image(wscreen, padded_shape)

        wconv = fft_image(wscreen_padded)
        wconv.data *= float(oversampling)**2
        # For the moment, ignore the polarisation and channel axes
        kernels.append(convert_image_to_kernel(wconv, oversampling,
                                               kernelwidth).data[0, 0, ...])
    
    # Now make a lookup table from row number of vis to the kernel
    kernel_indices = digitise(vis.w, wstep)
    assert numpy.max(kernel_indices) < len(kernels), "wabsmax %f wstep %f" % (wmaxabs, wstep)
    assert numpy.min(kernel_indices) >= 0, "wabsmax %f wstep %f" % (wmaxabs, wstep)
    return kernel_indices, kernels


def get_kernel_list(vis: Visibility, im: Image, **kwargs):
    """Get the list of kernels, one per visibility
    
    """
    
    shape = im.data.shape
    npixel = shape[3]
    cellsize = numpy.pi * im.wcs.wcs.cdelt[1] / 180.0
    
    wstep = get_parameter(kwargs, "wstep", 0.0)
    oversampling = get_parameter(kwargs, "oversampling", 8)
    padding = get_parameter(kwargs, "padding", 2)
    
    gcf, _ = anti_aliasing_calculate((padding * npixel, padding * npixel), oversampling)
    
    wabsmax = numpy.max(numpy.abs(vis.w))
    if wstep > 0.0 and wabsmax > 0.0:
        kernelname = 'wprojection'
        # wprojection needs a lot of commentary!
        log.debug("get_kernel_list: Using w projection with wstep = %f" % (wstep))

        # The field of view must be as padded! R_F is for reporting only so that
        # need not be padded.
        fov = cellsize * npixel * padding
        r_f = (cellsize * npixel / 2) ** 2 / abs(cellsize)
        log.debug("get_kernel_list: Fresnel number = %f" % (r_f))
 
        # Now calculate the maximum support for the w kernel
        kernelwidth = get_parameter(kwargs, "kernelwidth",
                                    (2 * int(round(numpy.sin(0.5 * fov) * npixel * wabsmax * cellsize))))
        kernelwidth = max(kernelwidth, 8)
        assert kernelwidth % 2 == 0
        log.debug("get_kernel_list: Maximum w kernel full width = %d pixels" % (kernelwidth))
        padded_shape = [im.shape[0], im.shape[1], im.shape[2] * padding, im.shape[3] * padding]

        remove_shift = get_parameter(kwargs, "remove_shift", True)
        padded_image = pad_image(im, padded_shape)
        kernel_list = w_kernel_list(vis, padded_image, oversampling=oversampling, wstep=wstep,
                                    kernelwidth=kernelwidth, remove_shift=remove_shift)
    else:
        kernelname = '2d'
        kernel_list = standard_kernel_list(vis, (padding * npixel, padding * npixel),
                                           oversampling=oversampling)
    
    return kernelname, gcf, kernel_list
