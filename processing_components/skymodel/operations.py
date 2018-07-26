"""Function to manage skymodels.

"""

import logging

from data_models.memory_data_models import SkyModel

from libs.image.operations import copy_image

from ..skycomponent.base import copy_skycomponent

log = logging.getLogger(__name__)

def copy_skymodel(sm):
    """ Copy a sky model
    
    """
    return SkyModel(components=[copy_skycomponent(comp) for comp in sm.components],
                    images=[copy_image(im) for im in sm.images],
                    fixed=sm.fixed)
