""" Radio interferometric calibration using the modelpartition algorithm and consensus optimisation.

This works as follows:

"""

import logging

from wrappers.arlexecute.calibration.modelpartition import solve_modelpartitions, create_modelpartitions
from wrappers.arlexecute.calibration.operations import copy_gaintable, create_gaintable_from_blockvisibility
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.skymodel.operations import copy_skymodel
from wrappers.arlexecute.visibility.base import copy_visibility

log = logging.getLogger(__name__)


def solve_modelpartition_consensus_list_arlexecute_workflow(vislist, skymodel_list, niter=10, coniter=10, tol=1e-8,
                                                       gain=0.25, **kwargs):
    """ Solve using modelpartition with consensus optimisation, dask.delayed wrapper

    Solve by iterating, performing E step and M step.

    :param vis_list: Initial visibility
    :param skymodel_list: List of sky models, one per vis
    :param kwargs:
    :return: A dask graph to calculate the individual data models and the residual visibility
    """
    model_partitions = [arlexecute.execute(create_modelpartitions, nout=2)(vislist[i], skymodel_list[i],**kwargs)
                        for i, _ in enumerate(vislist)]
    for iter in range(coniter):
        partition_results = [arlexecute.execute(solve_modelpartition, nout=2)(vislist[i], skymodel_list[i],
                                                                              niter=niter, tol=1e-8, gain=0.25,
                                                                              **kwargs)
                             for i, _ in enumerate(vislist)]
        skymodel_list = calculate_modelpartition_consensus_arlexecute_workflow(partition_results)
    
    return arlexecute.execute((skymodel_list)(skymodel_list))

def calculate_modelpartition_consensus_arlexecute_workflow(partition_results, **kwargs):
    """ Find consensus of models
    
    :param partition_results: List of (vis, skymodel)
    :param skymodel_list:
    :param kwargs:
    :return:

    """
    return skymodel_list
