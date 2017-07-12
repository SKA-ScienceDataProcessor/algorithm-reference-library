""" Functions for peeling known sources (represented as Skycomponents) from visibility sets.

"""

import collections
import logging

from arl.calibration.operations import apply_gaintable
from arl.calibration.solvers import solve_gaintable
from arl.data.data_models import BlockVisibility, Skycomponent
from arl.visibility.operations import copy_visibility
from arl.imaging.base import predict_skycomponent_blockvisibility

log = logging.getLogger(__name__)

def peel_skycomponent_blockvisibility(vis: BlockVisibility, sc: Skycomponent, remove=True) -> \
        BlockVisibility:
    """ Peel a collection of components.
    
    Sequentially solve the gain towards each Skycomponent and optionally remove from the visibility.

    :param params:
    :param vis: Visibility to be processed
    :param sc: Skycomponent or list of Skycomponents
    :returns: subtracted visibility and list of GainTables
    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis

    if not isinstance(sc, collections.Iterable):
        sc = [sc]

    gtlist = []
    for comp in sc:
        assert comp.shape == 'Point', "Cannot handle shape %s" % comp.shape
        
        modelvis = copy_visibility(vis)
        modelvis = predict_skycomponent_blockvisibility(modelvis, comp)
        gt = solve_gaintable(vis, modelvis, phase_only=False)
        modelvis = apply_gaintable(modelvis, gt)
        if remove:
            vis.data['vis'] -= modelvis.data['vis']
        gtlist.append(gt)
        
    return vis, gtlist