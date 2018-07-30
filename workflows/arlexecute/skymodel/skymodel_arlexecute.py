"""

"""
import logging

log = logging.getLogger(__name__)

from processing_components.skycomponent.base import copy_skycomponent
from processing_components.imaging.base import predict_skycomponent_visibility
from workflows.arlexecute.imaging.imaging_arlexecute import predict_arlexecute
from data_models.memory_data_models import SkyModel
from libs.image.operations import copy_image
from workflows.arlexecute.image.solvers import solve_image_arlexecute
from processing_components.visibility.visibility_fitting import fit_visibility


def predict_skymodel_visibility(vis, sm, **kwargs):
    """ Predict the visibility for a sky model.
    
    The skymodel is a collection of skycomponents and images
    
    :param vis:
    :param sm:
    :param kwargs:
    :return:
    """
    if sm.components is not None:
        vis = predict_skycomponent_visibility(vis, sm.components)
    if sm.images is not None:
        for im in sm.images:
            vis = predict_arlexecute(vis, im, **kwargs)
    
    return vis


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
        new_image = solve_image_arlexecute(vis, new_image, **kwargs)
        new_images.append(new_image)
    
    return SkyModel(components=new_comps, images=new_images)
