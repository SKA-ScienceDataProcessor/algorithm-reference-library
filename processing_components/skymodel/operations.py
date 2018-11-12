"""Function to manage skymodels.

"""

import logging

import numpy

from data_models.memory_data_models import SkyModel
from processing_library.image.operations import copy_image
from ..skycomponent.base import copy_skycomponent
from ..skycomponent.operations import filter_skycomponents_by_flux, insert_skycomponent
from ..visibility.visibility_fitting import fit_visibility

log = logging.getLogger(__name__)


def copy_skymodel(sm):
    """ Copy a sky model
    
    """
    return SkyModel(components=[copy_skycomponent(comp) for comp in sm.components],
                    images=[copy_image(im) for im in sm.images],
                    fixed=sm.fixed)


def split_skycomponents_to_skymodel(sc, model, flux_threshold=-numpy.inf):
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


def solve_skymodel(vis, skymodel, gain=0.1, **kwargs):
    """Fit a single skymodel to a visibility
    
    :param evis: Expected vis for this ssm
    :param modelpartition: scm element being fit i.e. (skymodel, gaintable) tuple
    :param gain: Gain in step
    :param method: 'fit' or 'sum'
    :param kwargs:
    :return: skycomponent
    """
    if skymodel.fixed:
        return skymodel
    
    new_comps = list()
    for comp in skymodel.components:
        new_comp = copy_skycomponent(comp)
        new_comp, _ = fit_visibility(vis, new_comp)
        new_comp.flux = gain * new_comp.flux + (1.0 - gain) * comp.flux
        new_comps.append(new_comp)
    
    new_images = list()
    for im in skymodel.images:
        new_image = copy_image(im)
        #        new_image = solve_image_arlexecute_workflow(vis, new_image, **kwargs)
        new_images.append(new_image)
    
    return SkyModel(components=new_comps, images=new_images)
