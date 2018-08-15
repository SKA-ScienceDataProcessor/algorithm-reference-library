"""

"""
import logging

log = logging.getLogger(__name__)

from wrappers.arlexecute.imaging.base import predict_skycomponent_visibility
from workflows.arlexecute.imaging.imaging_arlexecute import predict_arlexecute_workflow


def predict_skymodel_visibility_workflow(vis, sm, **kwargs):
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
            vis = predict_arlexecute_workflow(vis, im, **kwargs)
    
    return vis
