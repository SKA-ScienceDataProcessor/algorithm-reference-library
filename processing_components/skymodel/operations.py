"""Function to manage skymodels.

"""

import logging

import numpy
from astropy.wcs.utils import skycoord_to_pixel

from data_models.memory_data_models import SkyModel
from processing_library.image.operations import copy_image
from ..calibration.operations import copy_gaintable
from ..image.operations import smooth_image
from ..skycomponent.base import copy_skycomponent
from ..skycomponent.operations import filter_skycomponents_by_flux, insert_skycomponent

log = logging.getLogger(__name__)


def copy_skymodel(sm):
    """ Copy a sky model
    
    """
    if sm.components is not None:
        newcomps = [copy_skycomponent(comp) for comp in sm.components]
    else:
        newcomps = None
    
    if sm.image is not None:
        newimage = copy_image(sm.image)
    else:
        newimage = None
    
    if sm.mask is not None:
        newmask = copy_image(sm.mask)
    else:
        newmask = None
    
    if sm.gaintable is not None:
        newgt = copy_gaintable(sm.gaintable)
    else:
        newgt = None
    
    return SkyModel(components=newcomps, image=newimage, gaintable=newgt, mask=newmask,
                    fixed=sm.fixed)


def partition_skymodel_by_flux(sc, model, flux_threshold=-numpy.inf):
    """Partition skymodel according to flux
    
    :param sc:
    :param model:
    :param flux_threshold:
    :return:
    """
    brightsc = filter_skycomponents_by_flux(sc, flux_min=flux_threshold)
    weaksc = filter_skycomponents_by_flux(sc, flux_max=flux_threshold)
    log.info('Converted %d components into %d bright components and one image containing %d components'
             % (len(sc), len(brightsc), len(weaksc)))
    im = copy_image(model)
    im = insert_skycomponent(im, weaksc)
    return SkyModel(components=[copy_skycomponent(comp) for comp in brightsc],
                    image=copy_image(im), mask=None,
                    fixed=False)
