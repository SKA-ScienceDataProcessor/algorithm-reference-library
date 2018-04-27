""" Common functions converted to Dask.execute components. `Dask <http://dask.pydata.org/>`_ is a python-based flexible
parallel computing library for analytic computing. Dask.delayed can be used to wrap functions for deferred execution
thus allowing construction of components. For example, to build a graph for a major/minor cycle algorithm::

    model_imagelist = arlexecute.compute(create_image_from_visibility)(vt, npixel=512, cellsize=0.001, npol=1)
    solution_list = create_solve_image_list(vt, model_imagelist=model_imagelist, psf_list=psf_list,
                                            context='timeslice', algorithm='hogbom',
                                            niter=1000, fractional_threshold=0.1,
                                            threshold=1.0, nmajor=3, gain=0.1)
    solution_list.visualize()

The graph for one vis_list is executed as follows::

    solution_list[0].compute()
    
or if a Dask.distributed client is available:

    client.compute(solution_list)

As well as the specific components constructed by functions in this module, there are generic versions in the module
:mod:`libs.pipelines.generic_dask_lists`.

Construction of the components requires that the number of nodes (e.g. w slices or time-slices) be known at construction,
rather than execution. To counteract this, at run time, a given node should be able to act as a no-op. We use None
to denote a null node.

The actual imaging code executed eventually is specified by the context variable (see libs.imaging.imaging)context.
These are the same as executed in the imaging framework.

"""

from ..component_support.arlexecute import arlexecute
from ..calibration.calibration_control import calibrate_function
from ..calibration.operations import apply_gaintable
from ..visibility.gather_scatter import visibility_gather_channel
from ..visibility.operations import divide_visibility, integrate_visibility_by_channel


def calibrate_component(vis_list, model_vislist, calibration_context='TG', global_solution=True,
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
    
    def solve_and_apply(vis, modelvis=None):
        return calibrate_function(vis, modelvis, calibration_context=calibration_context, **kwargs)[0]
    
    if global_solution:
        point_vislist = [arlexecute.execute(divide_visibility, nout=len(vis_list))(vis_list[i],
                                                                                   model_vislist[i])
                         for i, _ in enumerate(vis_list)]
        global_point_vis_list = arlexecute.execute(visibility_gather_channel, nout=1)(point_vislist)
        global_point_vis_list = arlexecute.execute(integrate_visibility_by_channel, nout=1)(global_point_vis_list)
        # This is a global solution so we only compute one gain table
        _, gt_list = arlexecute.execute(solve_and_apply, pure=True, nout=2)(global_point_vis_list, **kwargs)
        return [arlexecute.execute(apply_gaintable, nout=len(vis_list))(v, gt_list, inverse=True)
                for v in vis_list]
    else:
        
        return [
            arlexecute.execute(solve_and_apply, nout=len(vis_list))(vis_list[i], model_vislist[i])
            for i, v in enumerate(vis_list)]
