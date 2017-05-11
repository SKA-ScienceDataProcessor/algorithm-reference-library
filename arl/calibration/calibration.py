""" Functions to solve for antenna/station gain

This uses an iterative substitution algorithm due to Larry D'Addario c 1980'ish. Used
in the original VLA Dec-10 Antsol.

For example::

    gtsol = solve_gaintable(vis, originalvis, phase_only=True, niter=niter, crosspol=False, tol=1e-6)
    vis = apply_gaintable(vis, gtsol, inverse=True)
 

"""

import numpy
import logging
import collections

from arl.data.data_models import Visibility, BlockVisibility, Skycomponent, Image
from arl.visibility.iterators import vis_timeslice_iter
from arl.visibility.operations import copy_visibility
from arl.fourier_transforms.ftprocessor_base import predict_skycomponent_blockvisibility, predict_2d, \
    predict_skycomponent_visibility, invert_2d, predict_2d
from arl.visibility.coalesce import convert_blockvisibility_to_visibility, decoalesce_visibility
from arl.calibration.operations import apply_gaintable
from arl.calibration.solvers import solve_gaintable

log = logging.getLogger(__name__)

def peel_skycomponent_blockvisibility(vis: BlockVisibility, sc: Skycomponent, remove=True) -> \
        BlockVisibility:
    """ Peel a collection of component.
    
    Sequentially solve the gain towards each Skycomponent and optionally remove from the visibility.

    :param params:
    :param vis: Visibility to be processed
    :param sc: Skycomponent or list of Skycomponents
    :returns: subtracted visibility and list of GainTables
    """
    # TODO: Implement peeling
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis

    if not isinstance(sc, collections.Iterable):
        sc = [sc]

    gtlist = []
    for comp in sc:
        assert comp.shape == 'Point', "Cannot handle shape %s" % comp.shape
        
        modelvis = copy_visibility(vis)
        modelvis = predict_skycomponent_blockvisibility(modelvis, comp)
        gt = solve_gaintable(vis, modelvis)
        modelvis = apply_gaintable(modelvis, gt)
        if remove:
            vis.data -= modelvis.data
        gtlist.append(gt)
        
    return vis, gtlist


def calibrate_blockvisibility(bvt: BlockVisibility, model=None, components=None,
                              predict=predict_2d, **kwargs):
    """ calibrate BlockVisibility with respect to model and optionally components

    :param bvt: BlockVisibility
    :param model: Model image
    :param components: Sky components
    :returns: Calibrated BlockVisibility

    """
    assert model is not None or components is not None, "calibration requires a model or skycomponents"
    
    if model is not None:
        vtpred = convert_blockvisibility_to_visibility(bvt)
        vtpred = predict(vtpred, model, **kwargs)
        bvtpred = decoalesce_visibility(vtpred)
        if components is not None:
            bvtpred = predict_skycomponent_blockvisibility(bvtpred, components)
    else:
        bvtpred = copy_visibility(bvt, zero=True)
        bvtpred = predict_skycomponent_blockvisibility(bvtpred, components)
    
    gt = solve_gaintable(bvt, bvtpred, **kwargs)
    return apply_gaintable(bvt, gt, **kwargs)


def calibrate_visibility(vt: Visibility, model=None, components=None, predict=predict_2d,
                         **kwargs):
    """ calibrate Visibility with respect to model and optionally components

    :param vt: Visibility
    :param model: Model image
    :param components: Sky components
    :returns: Calibrated visibility
    """
    assert model is not None or components is not None, "calibration requires a model or skycomponents"
    
    vtpred = copy_visibility(vt, zero=True)
    
    if model is not None:
        vtpred = predict(vtpred, model, **kwargs)
        if components is not None:
            vtpred = predict_skycomponent_visibility(vtpred, components)
    else:
        vtpred = predict_skycomponent_visibility(vtpred, components)
    
    bvt = decoalesce_visibility(vt)
    bvtpred = decoalesce_visibility(vtpred)
    gt = solve_gaintable(bvt, bvtpred, **kwargs)
    bvt = apply_gaintable(bvt, gt, **kwargs)
    return convert_blockvisibility_to_visibility(bvt)[0]

