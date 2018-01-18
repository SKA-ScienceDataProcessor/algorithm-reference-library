""" Functions for peeling known sources (represented as Skycomponents) from visibility sets.

"""

from typing import Union, List
import collections
import logging

from arl.calibration.operations import apply_gaintable
from arl.calibration.solvers import solve_gaintable
from arl.data.data_models import BlockVisibility, Skycomponent, GainTable
from arl.visibility.base import copy_visibility
from arl.imaging.base import predict_skycomponent_visibility

log = logging.getLogger(__name__)


def peel_skycomponent_blockvisibility(vis: BlockVisibility, sc: Union[Skycomponent, List[Skycomponent]],
                                      remove=True, **kwargs)\
        -> (BlockVisibility, List[GainTable]):
    """ Peel a collection of components.
    
    Sequentially solve the gain towards each Skycomponent and optionally remove the corrupted visibility from the
    observed visibility.

    :param params:
    :param vis: Visibility to be processed
    :param sc: Skycomponent or list of Skycomponents
    :return: subtracted visibility and list of GainTables
    """
    assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis

    if not isinstance(sc, collections.Iterable):
        sc = [sc]

    gtlist = []
    for comp in sc:
        assert comp.shape == 'Point', "Cannot handle shape %s" % comp.shape
        
        modelvis = copy_visibility(vis, zero=True)
        modelvis = predict_skycomponent_visibility(modelvis, comp)
        gt = solve_gaintable(vis, modelvis, phase_only=False, **kwargs)
        modelvis = apply_gaintable(modelvis, gt, **kwargs)
        if remove:
            vis.data['vis'] -= modelvis.data['vis']
        gtlist.append(gt)
        
    return vis, gtlist
