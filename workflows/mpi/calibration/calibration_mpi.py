"""

"""

from processing_components.calibration.calibration_control import calibrate_function
from processing_components.calibration.operations import apply_gaintable
from processing_components.visibility.gather_scatter import visibility_gather_channel
from processing_components.visibility.operations import divide_visibility, integrate_visibility_by_channel


import numpy
from mpi4py import MPI
import logging
log = logging.getLogger(__name__)


def calibrate_list_mpi_workflow(vis_list, model_vislist,
                                calibration_context='TG', global_solution=True,
                                comm=MPI.COMM_WORLD,
                                   **kwargs):
    """ Create a set of components for (optionally global) calibration of a list of visibilities

    If global solution is true then visibilities are gathered to a single visibility data set which is then
    self-calibrated. The resulting gaintable is then effectively scattered out for application to each visibility
    set. If global solution is false then the solutions are performed locally.

    :param vis_list: in rank0
    :param model_vislist: in rank0
    :param calibration_context: String giving terms to be calibrated e.g. 'TGB'
    :param global_solution: Solve for global gains
    :param kwargs: Parameters for functions in components
    :return:
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    log.debug('%d: In calibrate_list_mpi_workflow : %d elements in vis_list' % (rank,len(vis_list)))
    
    def solve_and_apply(vis, modelvis=None):
        return calibrate_function(vis, modelvis, calibration_context=calibration_context, **kwargs)[0]
    
    if global_solution:

        
        sub_vis_list= numpy.array_split(vis_list, size)
        sub_vis_list=comm.scatter(sub_vis_list,root=0)
        sub_model_vislist= numpy.array_split(model_vislist, size)
        sub_model_vislist=comm.scatter(sub_model_vislist,root=0)

        sub_point_vislist = [divide_visibility(sub_vis_list[i], sub_model_vislist[i])
                         for i, _ in enumerate(sub_vis_list)]
        point_vislist = comm.gather(sub_point_vislist,root=0)
        if rank==0:
            point_vislist =numpy.concatenate(point_vislist)

            global_point_vis_list = visibility_gather_channel(point_vislist)
            global_point_vis_list = integrate_visibility_by_channel(global_point_vis_list)
            # This is a global solution so we only compute one gain table
            _, gt_list = solve_and_apply(global_point_vis_list, **kwargs)

        gt_list=comm.bcast(gt_list,root=0)
        
        sub_result = [apply_gaintable(v, gt_list, inverse=True)
                for v in sub_vis_list]
        result = comm.gather(sub_result,root=0)
        if rank==0:
            result = numpy.concatenate(result)
        else:
            result = list()
        return result
    else:
        sub_vis_list= numpy.array_split(vis_list, size)
        sub_vis_list=comm.scatter(sub_vis_list,root=0)
        sub_model_vislist= numpy.array_split(model_vislist, size)
        sub_model_vislist=comm.scatter(sub_model_vislist,root=0)
        
        sub_result = [solve_and_apply(sub_vis_list[i], sub_model_vislist[i])
                      for i, v in enumerate(sub_vis_list)]
        result= comm.gather(sub_result,root=0)
        if rank==0:
            result=numpy.concatenate(result)
        else:
            result=list()
        return result
