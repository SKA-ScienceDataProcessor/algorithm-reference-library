""" Radio interferometric calibration using an expectation maximisation algorithm

See the SDP document "Model Partition Calibration View Packet"

In this code:

- A single model parition is taken to be a list composed of (skymodel, gaintable) tuples.

- The E step for a specific model partition is the sum of the partition data model and the discrepancy between the
    observed data and the summed (over all partitions) data models.


- The M step for a specific partition is the optimisation of the model partition given the model partition. This
    involves fitting a skycomponent and fitting for the gain phases.

"""

import logging

from data_models.memory_data_models import BlockVisibility
from modelpartition_arlexecute import modelpartition_list_expectation_step, modelpartition_list_expectation_all, \
    modelpartition_list_maximisation_step
from processing_components.calibration.operations import copy_gaintable, create_gaintable_from_blockvisibility, qa_gaintable
from processing_components.skymodel.operations import copy_skymodel
from processing_components.visibility.operations import copy_visibility

log = logging.getLogger(__name__)


def create_modelpartition(vis: BlockVisibility, skymodels, **kwargs):
    """Create a set of associations between skymodel and gaintable

    :param vis: BlockVisibility to process
    :param skymodels: List of skyModels
    :param kwargs:
    :return:
    """
    gt = create_gaintable_from_blockvisibility(vis, **kwargs)
    return [(copy_skymodel(sm), copy_gaintable(gt)) for sm in skymodels]


def solve_modelpartition(vis, skymodels, niter=10, tol=1e-8, gain=0.25, **kwargs):
    """ Solve for model partitions

    Solve by iterating, performing E step and M step.

    :param vis: Initial visibility
    :param components: Initial components to be used
    :param gaintables: Initial gain tables to be used
    :param kwargs:
    :return: The individual data models and the residual visibility
    """
    model_partition = create_modelpartition(vis, skymodels=skymodels, **kwargs)
    
    for iter in range(niter):
        new_modelpartitions = list()
        evis_all = modelpartition_list_expectation_all(vis, model_partition)
        log.debug("solve_modelpartition: Iteration %d" % (iter))
        for window_index, csm in enumerate(model_partition):
            evis = modelpartition_list_expectation_step(vis, evis_all, csm, gain=gain, **kwargs)
            new_csm = modelpartition_list_maximisation_step(evis, csm, **kwargs)
            new_modelpartitions.append((new_csm[0], new_csm[1]))
            
            flux = new_csm[0].components[0].flux[0, 0]
            qa = qa_gaintable(new_csm[1])
            residual = qa.data['residual']
            rms_phase = qa.data['rms-phase']
            log.debug("solve_modelpartition:\t Window %d, flux %s, residual %.3f, rms phase %.3f" % (window_index,
                                                                                                     str(flux),
                                                                                                     residual,
                                                                                                     rms_phase))
        
        model_partition = [(copy_skymodel(csm[0]), copy_gaintable(csm[1])) for csm in new_modelpartitions]
    
    residual_vis = copy_visibility(vis)
    final_vis = modelpartition_list_expectation_all(vis, model_partition)
    residual_vis.data['vis'][...] = vis.data['vis'][...] - final_vis.data['vis'][...]
    return model_partition, residual_vis


