"""

"""

import numpy

from wrappers.serial.calibration.calibration_control import calibrate_function, apply_calibration_function, \
    solve_calibrate_function
from wrappers.serial.visibility.coalesce import convert_visibility_to_blockvisibility
from wrappers.serial.visibility.gather_scatter import visibility_gather_channel
from wrappers.serial.visibility.operations import integrate_visibility_by_channel, \
    divide_visibility


def calibrate_list_serial_workflow(vis_list, model_vislist, calibration_context='TG', global_solution=True,
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
            print("No solution in solve")
            return None
    
    def solve_and_apply(vis, modelvis=None):
        if modelvis is None or numpy.max(numpy.abs(modelvis.vis)) > 0.0:
            # Returns the block visibility and the gaintable
            return calibrate_function(vis, modelvis, calibration_context=calibration_context, **kwargs)
        else:
            print("No solution in solve and apply")
            return vis, None
    
    def apply(vis, gt):
        # Returns just the block visibility
        return apply_calibration_function(vis, gt, calibration_context=calibration_context, **kwargs)
    
    if global_solution:
        point_vislist = [convert_visibility_to_blockvisibility(v) for v in vis_list]
        point_modelvislist = [convert_visibility_to_blockvisibility(mv)
                              for mv in model_vislist]
        point_vislist = [divide_visibility(point_vislist[i], point_modelvislist[i])
                         for i, _ in enumerate(point_vislist)]
        point_vislist = [convert_visibility_to_blockvisibility(pv)
                         for pv in point_vislist]
        global_point_vis_list = visibility_gather_channel(point_vislist)
        global_point_vis_list = integrate_visibility_by_channel(global_point_vis_list)
        # This is a global solution so we only compute one gain table
        gt = solve(global_point_vis_list)
        return [apply(v, gt) for v in vis_list], gt
    else:
        
        result = [solve_and_apply(vis_list[i], model_vislist[i])
                  for i, v in enumerate(vis_list)]
        # Return list of BVs and list of gaintables
        return [r[0] for r in result], [r[1] for r in result]
