"""

"""

import numpy

from data_models.data_model_helpers import GainTable
from wrappers.mpi.calibration.calibration_control import calibrate_function, apply_calibration_function, \
    solve_calibrate_function
from wrappers.mpi.visibility.coalesce import convert_visibility_to_blockvisibility
from wrappers.mpi.visibility.gather_scatter import visibility_gather_channel
from wrappers.mpi.visibility.operations import integrate_visibility_by_channel, \
    divide_visibility

from mpi4py import MPI
import logging
log = logging.getLogger(__name__)

def calibrate_list_mpi_workflow(vis_list, model_vislist, calibration_context='TG', global_solution=True,
                                comm=MPI.COMM_WORLD,
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
    rank = comm.Get_rank()
    size = comm.Get_size()
    log.debug('%d: In calibrate_list_mpi_workflow : %d elements in vis_list' % (rank,len(vis_list)))
    
    def solve(vis, modelvis=None):
        return solve_calibrate_function(vis, modelvis, calibration_context=calibration_context, **kwargs)
    
    def apply(vis, gt):
        assert gt is not None
        return apply_calibration_function(vis, gt, calibration_context=calibration_context, **kwargs)
    
    sub_vis_list= numpy.array_split(vis_list, size)
    sub_vis_list=comm.scatter(sub_vis_list,root=0)
    sub_model_vislist= numpy.array_split(model_vislist, size)
    sub_model_vislist=comm.scatter(sub_model_vislist,root=0)
    
    if global_solution:
        sub_point_vislist = [convert_visibility_to_blockvisibility(v) for v in
                             sub_vis_list]
        sub_point_modelvislist = [convert_visibility_to_blockvisibility(mv)
                              for mv in sub_model_vislist]
        sub_point_vislist = [divide_visibility(sub_point_vislist[i], sub_point_modelvislist[i])
                         for i, _ in enumerate(sub_point_vislist)]
        point_vislist=comm.gather(sub_point_vislist,root=0)
        if rank==0:
            point_vislist =numpy.concatenate(point_vislist)

            global_point_vis_list = visibility_gather_channel(point_vislist)
            global_point_vis_list = integrate_visibility_by_channel(global_point_vis_list)
            # This is a global solution so we only compute one gain table
            gt_list = [solve(global_point_vis_list)]
        gt_list=comm.bcast(gt_list,root=0)
        
        sub_result = [apply(v, gt_list[0]) for v in sub_vis_list]
        result = comm.gather(sub_result,root=0)
        if rank==0:
            result = numpy.concatenate(result)
        else:
            result = list()
    else:
        sub_gt_list = [solve(v, sub_model_vislist[i])
                   for i, v in enumerate(sub_vis_list)]
        sub_result = [apply(v, sub_gt_list[i]) for i, v in enumerate(sub_vis_list)]
        result = comm.gather(sub_result,root=0)
        gt_list = comm.gather(sub_gt_list,root=0)
        if rank==0:
            result = numpy.concatenate(result)
            gt_list = numpy.concatenate(gt_list)
        else:
            result = list()
            gt_list = list()
    return result, gt_list
