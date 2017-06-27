"""
Functions that aid definition of fourier transform processing.
"""

import astropy.constants as constants

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
        assert min(vfrequencymap) >= 0, "Invalid frequency map: visibility channel < 0"


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
        
        assert min(vfrequencymap) >= 0, "Invalid frequency map: image channel < 0"
        assert max(vfrequencymap) < im.shape[0], "Invalid frequency map: image channel > number image channels"
    
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


def get_uvw_map(vis, im, **kwargs):
    """ Get the generators that map channels uvw to pixels

    """
    # Transform parameters
    padding = get_parameter(kwargs, "padding", 2)
    
    # Model image information
    inchan, inpol, ny, nx = im.data.shape
    shape = (1, int(round(padding * ny)), int(round(padding * nx)))
    # UV sampling information
    uvwscale = numpy.zeros([3])
    uvwscale[0:2] = im.wcs.wcs.cdelt[0:2] * numpy.pi / 180.0
    assert uvwscale[0] != 0.0, "Error in uv scaling"
    fov = int(round(padding * nx)) * numpy.abs(uvwscale[0])
    
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


def get_kernel_list(vis: Visibility, im, **kwargs):
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
        log.debug("get_kernel_list: Using wprojection kernel")
        wmax = numpy.max(numpy.abs(vis.w))
        assert wmax > 0, "Maximum w must be > 0.0"
        
        # The field of view must be as padded!
        fov = cellsize * npixel * padding
        r_f = (cellsize * npixel / 2) ** 2 / abs(cellsize)
        log.debug("get_kernel_list: Fresnel number = %f" % (r_f))
        delA = get_parameter(kwargs, 'wloss', 0.02)
        
        advice = advise_wide_field(vis, delA)
        wstep = get_parameter(kwargs, "wstep", advice['w_sampling_primary_beam'])
        
        log.debug("get_kernel_list: Using w projection with wstep = %f" % (wstep))
        
        # Now calculate the maximum support for the w kernel
        npixel_kernel = get_parameter(kwargs, "kernelwidth", (2 * int(round(numpy.sin(0.5 * fov) * npixel/4.0))))
        assert npixel_kernel % 2 == 0
        log.debug("get_kernel_list: Maximum w kernel full width = %d pixels" % (npixel_kernel))
        kernel_list = w_kernel_list(vis, (npixel, npixel), fov, wstep=wstep,
                                    npixel_kernel=npixel_kernel, oversampling=oversampling)
    else:
        kernelname = '2d'
        kernel_list = standard_kernel_list(vis, (padding * npixel, padding * npixel), oversampling=8, support=3)
    
    return kernelname, gcf, kernel_list

def advise_wide_field(vis, delA=0.02, oversampling_synthesised_beam=3.0, guard_band_image=6.0, facets=1.0):
    """ Advise on parameters for wide field imaging.
    
    For example::
    
        advice = advise_wide_field(vis, delA)
        wstep = get_parameter(kwargs, "wstep", advice['w_sampling_primary_beam'])

    
    :param vis:
    :param delA: Allowed coherence loss (def: 0.02)
    :param oversampling_synthesised_beam: Oversampling of the synthesized beam (def: 3.0)
    :param guard_band_image: Number of primary beam half-widths-to-half-maximum to image (def: 6)
    :returns: dict of advice
    """
    maximum_baseline = numpy.max(numpy.abs(vis.uvw)) # Wavelengths
    log.info("advise_wide_field: Maximum baseline %.1f (wavelengths)" % (maximum_baseline))
    
    diameter = numpy.min(vis.configuration.diameter)
    log.info("advise_wide_field: Station/antenna diameter %.1f (meters)" % (diameter))

    wavelength = constants.c.to('m/s').value / numpy.min(vis.frequency)
    log.info("advise_wide_field: Maximum wavelength %.3f (meters)" %(wavelength))

    primary_beam_fov = wavelength / diameter
    log.info("advise_wide_field: Primary beam %s" % (rad_and_deg(primary_beam_fov)))

    image_fov = primary_beam_fov * guard_band_image
    log.info("advise_wide_field: Image field of view %s" % (rad_and_deg(image_fov)))

    facet_fov = primary_beam_fov * guard_band_image / facets
    log.info("advise_wide_field: Facet field of view %s" % (rad_and_deg(facet_fov)))

    synthesized_beam = 1.0 / (maximum_baseline)
    log.info("advise_wide_field: Synthesized beam %s" % (rad_and_deg(synthesized_beam)))

    cellsize = synthesized_beam/oversampling_synthesised_beam
    log.info("advise_wide_field: Cellsize %s" % (rad_and_deg(cellsize)))

    npixels = int(round(image_fov/cellsize))
    log.info("advice_wide_field: Npixels per side = %d" % (npixels))

    # Following equation is from Cornwell, Humphreys, and Voronkov (2012) (equation 24)
    # We will assume that the constraint holds at one quarter the entire FOV i.e. that
    # the full field of view includes the entire primary beam

    w_sampling_image = numpy.sqrt(2.0 * delA) / (numpy.pi * image_fov ** 2)
    log.info("advice_wide_field: W sampling for full image = %.1f (wavelengths)" % (w_sampling_image))

    w_sampling_facet = numpy.sqrt(2.0 * delA) / (numpy.pi * facet_fov ** 2)
    log.info("advice_wide_field: W sampling for facet = %.1f (wavelengths)" % (w_sampling_image))

    w_sampling_primary_beam = numpy.sqrt(2.0 * delA) / (numpy.pi * primary_beam_fov ** 2)
    log.info("advice_wide_field: W sampling for primary beam = %.1f (wavelengths)" % (w_sampling_primary_beam))

    time_sampling_image =  86400.0 * w_sampling_image / (numpy.pi * maximum_baseline)
    log.info("advice_wide_field: Time sampling for full image = %.1f (s)" % (time_sampling_image))

    time_sampling_facet =  86400.0 * w_sampling_facet / (numpy.pi * maximum_baseline)
    log.info("advice_wide_field: Time sampling for facet = %.1f (s)" % (time_sampling_facet))

    time_sampling_primary_beam =  86400.0 * w_sampling_primary_beam / (numpy.pi * maximum_baseline)
    log.info("advice_wide_field: Time sampling for primary beam = %.1f (s)" % (time_sampling_primary_beam))

    freq_sampling_image =  numpy.max(vis.frequency) * w_sampling_image / (numpy.pi * maximum_baseline)
    log.info("advice_wide_field: Frequency sampling for full image = %.1f (Hz)" % (freq_sampling_image))

    freq_sampling_facet =  numpy.max(vis.frequency) * w_sampling_facet / (numpy.pi * maximum_baseline)
    log.info("advice_wide_field: Frequency sampling for facet = %.1f (Hz)" % (freq_sampling_facet))

    freq_sampling_primary_beam =  numpy.max(vis.frequency) * w_sampling_primary_beam / (numpy.pi * maximum_baseline)
    log.info("advice_wide_field: Frequency sampling for primary beam = %.1f (Hz)" % (freq_sampling_primary_beam))

    return locals()

def rad_and_deg(x):
    """ Stringify x in radian and degress forms
    
    """
    return "%.6f (rad) %.3f (deg)" % (x, 180.0 * x / numpy.pi)

