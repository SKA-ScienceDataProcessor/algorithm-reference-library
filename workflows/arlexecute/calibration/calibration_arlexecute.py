"""

"""

import numpy

from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.calibration.calibration_control import calibrate_function, apply_calibration_function, \
    solve_calibrate_function
from wrappers.arlexecute.visibility.coalesce import convert_visibility_to_blockvisibility
from wrappers.arlexecute.visibility.gather_scatter import visibility_gather_channel
from wrappers.arlexecute.visibility.operations import integrate_visibility_by_channel, \
    divide_visibility


def calibrate_list_arlexecute_workflow(vis_list, model_vislist, calibration_context='TG', global_solution=True,
                                       **kwargs):
    """ Create a set of components for (optionally global) calibration of a list of visibilities

    If global solution is true then visibilities are gathered to a single visibility data set which is then
    self-calibrated. The resulting gaintable is then effectively scattered out for application to each visibility
    set. If global solution is false then the solutions are performed locally.

    :param vis_list:
    :param model_vislist:
    :param calibration_context: String giving terms to be calibrated e.g. 'TGB'
    :param global_solution: Solve for global gains
    :param kwargs: Parameters for functions in components
    :return:
    """
    
    def solve(vis, modelvis=None):
        if modelvis is None or numpy.max(numpy.abs(modelvis.vis)) > 0.0:
            # Returns the gaintables
            return solve_calibrate_function(vis, modelvis, calibration_context=calibration_context, **kwargs)
        else:
            return None
    
    def apply(vis, gt):
        # Returns just the block visibility
        return apply_calibration_function(vis, gt, calibration_context=calibration_context, **kwargs)
    
    point_vislist = [arlexecute.execute(convert_visibility_to_blockvisibility, nout=1)(v) for v in vis_list]
    point_modelvislist = [arlexecute.execute(convert_visibility_to_blockvisibility, nout=1)(mv)
                          for mv in model_vislist]
    point_vislist = [arlexecute.execute(divide_visibility, nout=1)(point_vislist[i], point_modelvislist[i])
                     for i, _ in enumerate(point_vislist)]
    point_vislist = [arlexecute.execute(convert_visibility_to_blockvisibility, nout=1)(pv)
                         for pv in point_vislist]
    if global_solution:
        global_point_vis_list = arlexecute.execute(visibility_gather_channel, nout=1)(point_vislist)
        global_point_vis_list = arlexecute.execute(integrate_visibility_by_channel, nout=1)(global_point_vis_list)
        # This is a global solution so we only compute one gain table
        gt_list = arlexecute.execute(solve, pure=True, nout=1)(global_point_vis_list)
        return [arlexecute.execute(apply, nout=1)(v, gt_list) for v in vis_list], gt_list
    else:
        gt_list = [arlexecute.execute(solve, pure=True, nout=1)(v) for i, v in enumerate(point_vislist)]
        return [arlexecute.execute(apply)(v, gt_list[i]) for i, v in enumerate(vis_list)], gt_list
