"""Function to manage skymodels.

"""

import logging

import numpy

from data_models.memory_data_models import SkyModel, GainTable
from processing_library.image.operations import copy_image
from ..skycomponent.base import copy_skycomponent
from ..calibration.operations import copy_gaintable
from ..skycomponent.operations import filter_skycomponents_by_flux, insert_skycomponent
from ..visibility.visibility_fitting import fit_visibility

log = logging.getLogger(__name__)


def copy_skymodel(sm):
    """ Copy a sky model
    
    """
    return SkyModel(components=[copy_skycomponent(comp) for comp in sm.components],
                    images=[copy_image(im) for im in sm.images],
                    gaintables=[copy_gaintable(gt) for gt in sm.gaintables],
                    fixed=sm.fixed)


def partition_skymodel_by_flux(sc, model, flux_threshold=-numpy.inf):
    """
    
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
                    images=[copy_image(im)],
                    fixed=False)
