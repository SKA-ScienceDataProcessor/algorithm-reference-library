# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

import numpy

from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from arl.data_models import *
from arl.parameters import *

import logging
log = logging.getLogger("arl.skymodel_operations")

def create_skycomponent(direction: SkyCoord, flux: numpy.array, frequency: numpy.array, shape: str = 'Point',
                        param: dict = None, name: str = ''):
    """ A single SkyComponent with direction, flux, shape, and params for the shape

    :param direction:
    :type SkyCoord:
    :param flux:
    :type numpy.array:
    :param frequency:
    :type numpy.array:
    :param shape: 'Point' or 'Gaussian'
    :type str:
    :param name:
    :type str:
    :returns: SkyComponent
    """
    sc = SkyComponent()
    sc.direction = direction
    sc.frequency = frequency
    sc.name = name
    sc.flux = numpy.array(flux)
    sc.shape = shape
    sc.params = param
    sc.name = name
    return sc


def find_skycomponent(im: Image, params={}):
    """ Find components in Image, return SkyComponent, just find the peak for now

    :param im: Image to be searched
    :type Image:
    :returns: SkyComponent
    """
    # TODO: Implement full image fitting of components
    log_parameters(params)
    log.debug("point_source_find: Finding components in Image")
    
    # Beware: The index sequencing is opposite in wcs and Python!
    locpeak = numpy.array(numpy.unravel_index((numpy.abs(im.data)).argmax(), im.data.shape))
    log.debug("point_source_find: Found peak at pixel coordinates %s" % str(locpeak))
    w = im.wcs.sub(['longitude', 'latitude'])
    sc = pixel_to_skycoord(locpeak[3], locpeak[2], im.wcs, 0, 'wcs')
    log.debug("point_source_find: Found peak at world coordinates %s" % str(sc))
    flux = im.data[:, :, locpeak[2], locpeak[3]]
    log.debug("point_source_find: Flux is %s" % flux)
    # We also need the frequency values
    w = im.wcs.sub(['spectral'])
    frequency = w.wcs_pix2world(range(im.data.shape[0]), 1)
    return create_skycomponent(direction=sc, flux=flux, frequency=frequency, shape='point')


def fit_skycomponent(im: Image, sc: SkyCoord, params={}):
    """ Find flux at a given direction, return SkyComponent

    :param im:
    :type Image:
    :param sc:
    :type SkyCoord:
    :returns: SkyComponent

    """
    log_parameters(params)
    log.debug("find_flux_at_direction: Extracting flux at world coordinates %s" % str(sc))
    w = im.wcs.sub(['longitude', 'latitude'])
    pixloc = skycoord_to_pixel(sc, im.wcs, 0, 'wcs')
    log.debug("find_flux_at_direction: Extracting flux at pixel coordinates %d %d" % (pixloc[0], pixloc[1]))
    flux = im.data[:, :, int(pixloc[1] + 0.5), int(pixloc[0] + 0.5)]
    log.debug("find_flux_at_direction: Flux is %s" % flux)
    
    # We also need the frequency values
    w = im.wcs.sub(['spectral'])
    frequency = w.wcs_pix2world(range(im.data.shape[0]), 0)
    
    return create_skycomponent(direction=sc, flux=flux, frequency=frequency, shape='point')


def add_skymodels(sm1: SkyModel, sm2: SkyModel):
    """ Add two sky models together
    
    :param sm1:
    :type SkyModel:
    :param sm2:
    :type SkyModel:
    :returns: SkyModel
    """
    fsm = SkyModel()
    fsm.images = [sm1.images, sm2.images]
    fsm.components = [sm1.components, sm2.components]
    return fsm


def create_skymodel_from_image(im: Image):
    """ Create a skymodel from an image or image
    
    :param im:
    :type Image:
    :returns: SkyModel
    """
    sm = SkyModel()
    sm.images.append(im)
    return sm


def add_image_to_skymodel(sm: SkyModel, im: Image):
    """Add images to a sky model
    
    :param sm:
    :type SkyModel:
    :param im:
    :type Image:
    :returns: SkyModel
    """
    sm.images.append(im)
    return sm


def create_skymodel_from_component(comp: SkyComponent):
    """Create sky model from component
    
    :param comp:
    :type SkyComponent:
    :returns: SkyModel
    """
    sm = SkyModel()
    sm.components.append(comp)
    return sm


def add_component_to_skymodel(sm: SkyModel, comp: SkyComponent):
    """Add Component to a sky model
    
    :param sm:
    :type SkyModel:
    :param comp:
    :type SkyComponent:
    :returns: SkyModel
   """
    sm.components.append(comp)
    return sm
