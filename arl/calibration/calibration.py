""" Functions to solve for antenna/station gain

This uses an iterative substitution algorithm due to Larry D'Addario c 1980'ish. Used
in the original VLA Dec-10 Antsol.

For example::

    gtsol = solve_gaintable(vis, originalvis, phase_only=True, niter=niter, crosspol=False, tol=1e-6)
    vis = apply_gaintable(vis, gtsol, inverse=True)


"""

import logging

from arl.calibration.operations import apply_gaintable
from arl.calibration.solvers import solve_gaintable
from arl.data.data_models import Visibility, BlockVisibility, Image
from arl.visibility.coalesce import convert_blockvisibility_to_visibility, decoalesce_visibility
from arl.visibility.base import copy_visibility
from arl.imaging import predict_skycomponent_visibility, predict_2d
from arl.data.parameters import get_parameter

log = logging.getLogger(__name__)


def calibrate_blockvisibility(bvt: BlockVisibility, model: Image=None, components=None,
                              predict=predict_2d, **kwargs) -> BlockVisibility:
    """ calibrate BlockVisibility with respect to model and optionally components

    :param bvt: BlockVisibility
    :param model: Model image
    :param components: Sky components
    :return: Calibrated BlockVisibility

    """
    assert model is not None or components is not None, "calibration requires a model or skycomponents"

    if model is not None:
        vtpred = convert_blockvisibility_to_visibility(bvt)
        vtpred = predict(vtpred, model, **kwargs)
        bvtpred = decoalesce_visibility(vtpred)
        if components is not None:
            bvtpred = predict_skycomponent_visibility(bvtpred, components)
    else:
        bvtpred = copy_visibility(bvt, zero=True)
        bvtpred = predict_skycomponent_visibility(bvtpred, components)

    gt = solve_gaintable(bvt, bvtpred, **kwargs)
    return apply_gaintable(bvt, gt, inverse=get_parameter(kwargs, "inverse", False))


def calibrate_visibility(vt: Visibility, model: Image =None, components=None, predict=predict_2d,
                         **kwargs) -> Visibility:
    """ calibrate Visibility with respect to model and optionally components

    :param vt: Visibility
    :param model: Model image
    :param components: Sky components
    :return: Calibrated visibility
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
    bvt = apply_gaintable(bvt, gt, inverse=get_parameter(kwargs, "inverse", False))
    return convert_blockvisibility_to_visibility(bvt)
