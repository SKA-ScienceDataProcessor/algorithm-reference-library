# Tim Cornwell <realtimcornwell@gmail.com>
""" Synthesis imaging algorithms, top level

"""

from astropy import units as units
from astropy.constants import c
from astropy import wcs

from arl.coordinate_support import simulate_point, skycoord_to_lmn
from arl.data_models import *
from arl.image_operations import create_empty_image_like
from arl.ftprocessor import *
from arl.parameters import *

log = logging.getLogger("arl.fourier_transforms")

"""
Functions that perform imaging i.e. conversion of an Image to/from a Visibility
"""


def _create_wcs_from_visibility(vis, params=None):
    """Make a world coordinate system from params and Visibility

    :param vis:
    :param params: keyword=value parameters
    :returns: WCS
    """
    if params is None:
        params = {}
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
    uvmax = (numpy.abs(vis.data['uvw']).max() * reffrequency / c).value
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


def invert_visibility(vis, model, params=None):
    """Invert to make dirty Image and PSF
    
    This is the top level invert routine. It's basically a switch yard for the low level routines

    :param params:
    :param vis:
    :param model: Template model
    :returns: (dirty image, psf)
    """
    if params is None:
        params = {}
    log_parameters(params)
    log.debug("invert_visibility: Inverting Visibility to make dirty and psf")
    shape, reffrequency, cellsize, image_wcs, imagecentre = _create_wcs_from_visibility(vis, params=params)

    nchan, npol, _, npixel = shape
    field_of_view = npixel * cellsize
    
    log.debug("invert_visibility: Specified npixel=%d, cellsize = %f rad, FOV = %f rad" %
              (npixel, cellsize, field_of_view))
    
    dirty = create_empty_image_like(model)
    psf = create_empty_image_like(model)
    sumofweights = 0.0
    
    spectral_mode = get_parameter(params, 'spectral_mode', 'channel')
    log.debug('invert_visibility: spectral mode is %s' % spectral_mode)
    
    if spectral_mode == 'channel':
        dirty, psf, sumofweights = invert_2d(vis, dirty, psf, sumofweights, params=params)
        assert sumofweights > 0.0, "No data gridded"
    else:
        raise NotImplementedError("mode %s not supported" % spectral_mode)
    
    log.debug("invert_visibility: Finished making dirty and psf")
    
    return dirty, psf, sumofweights

