""" Functions to solve for antenna/station gain

This uses an iterative substitution algorithm due to Larry D'Addario c 1980'ish. Used
in the original VLA Dec-10 Antsol.

For example::

    gtsol = solve_gaintable(vis, originalvis, phase_only=True, niter=niter, crosspol=False, tol=1e-6)
    vis = apply_gaintable(vis, gtsol, inverse=True)


"""

__all__ = ['solve_gaintable']

import logging

import numpy

from data_models import BlockVisibility, GainTable, assert_vis_gt_compatible
from processing_components.calibration.operations import create_gaintable_from_blockvisibility
from processing_components.visibility import create_visibility_from_rows, divide_visibility
from processing_library.calibration.solvers import solve_from_X

log = logging.getLogger(__name__)

def solve_gaintable(vis: BlockVisibility, modelvis: BlockVisibility = None, gt=None, phase_only=True, niter=30,
                    tol=1e-8, crosspol=False, normalise_gains=True, **kwargs) -> GainTable:
    """Solve a gain table by fitting an observed visibility to a model visibility
    
    If modelvis is None, a point source model is assumed.

    :param vis: BlockVisibility containing the observed data_models
    :param modelvis: BlockVisibility containing the visibility predicted by a model
    :param gt: Existing gaintable
    :param phase_only: Solve only for the phases (default=True)
    :param niter: Number of iterations (default 30)
    :param tol: Iteration stops when the fractional change in the gain solution is below this tolerance
    :param crosspol: Do solutions including cross polarisations i.e. XY, YX or RL, LR
    :return: GainTable containing solution

    """
    assert isinstance(vis, BlockVisibility), vis
    if modelvis is not None:
        assert isinstance(modelvis, BlockVisibility), modelvis
        assert numpy.max(numpy.abs(modelvis.vis)) > 0.0, "Model visibility is zero"
    
    if phase_only:
        log.debug('solve_gaintable: Solving for phase only')
    else:
        log.debug('solve_gaintable: Solving for complex gain')
    
    if gt is None:
        log.debug("solve_gaintable: creating new gaintable")
        gt = create_gaintable_from_blockvisibility(vis, **kwargs)
    else:
        log.debug("solve_gaintable: starting from existing gaintable")
    
    for row in range(gt.ntimes):
        vis_rows = numpy.abs(vis.time - gt.time[row]) < gt.interval[row] / 2.0
        if numpy.sum(vis_rows) > 0:
            subvis = create_visibility_from_rows(vis, vis_rows)
            if modelvis is not None:
                model_subvis = create_visibility_from_rows(modelvis, vis_rows)
                pointvis = divide_visibility(subvis, model_subvis)
                x = numpy.sum(pointvis.vis * pointvis.weight, axis=0)
                xwt = numpy.sum(pointvis.weight, axis=0)
            else:
                x = numpy.sum(subvis.vis * subvis.weight, axis=0)
                xwt = numpy.sum(subvis.weight, axis=0)
            
            mask = numpy.abs(xwt) > 0.0
            x_shape = x.shape
            x[mask] = x[mask] / xwt[mask]
            x[~mask] = 0.0
            x = x.reshape(x_shape)
            
            gt = solve_from_X(gt, x, xwt, row, crosspol, niter, phase_only,
                              tol, npol=vis.polarisation_frame.npol)
            if normalise_gains and not phase_only:
                gabs = numpy.average(numpy.abs(gt.data['gain'][row]))
                gt.data['gain'][row] /= gabs
    
    assert isinstance(gt, GainTable), "gt is not a GainTable: %r" % gt
    
    assert_vis_gt_compatible(vis, gt)
    
    return gt
