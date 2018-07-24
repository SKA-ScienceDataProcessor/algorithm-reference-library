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

import numpy

from data_models.memory_data_models import BlockVisibility
from processing_components.calibration.calibration import solve_gaintable
from processing_components.calibration.operations import copy_gaintable, apply_gaintable, \
    create_gaintable_from_blockvisibility, qa_gaintable
from processing_components.skymodel.operations import copy_skymodel
from processing_components.skymodel.operations import predict_skymodel_visibility, solve_skymodel
from processing_components.visibility.coalesce import convert_blockvisibility_to_visibility
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


def modelpartition_fit_skymodel(vis, modelpartition, gain=0.1, **kwargs):
    """Fit a single skymodel to a visibility

    :param evis: Expected vis for this ssm
    :param modelpartition: scm element being fit i.e. (skymodel, gaintable) tuple
    :param gain: Gain in step
    :param kwargs:
    :return: skymodel
    """
    if modelpartition[0].fixed:
        return modelpartition[0]
    else:
        cvis = convert_blockvisibility_to_visibility(vis)
        return solve_skymodel(cvis, modelpartition[0], **kwargs)


def modelpartition_fit_gaintable(evis, modelpartition, gain=0.1, niter=3, tol=1e-3, **kwargs):
    """Fit a gaintable to a visibility
    
    This is the update to the gain part of the window

    :param evis: Expected vis for this ssm
    :param modelpartition: csm element being fit
    :param gain: Gain in step
    :param niter: Number of iterations
    :param kwargs: Gaintable
    """
    previous_gt = copy_gaintable(modelpartition[1])
    gt = copy_gaintable(modelpartition[1])
    model_vis = copy_visibility(evis, zero=True)
    model_vis = predict_skymodel_visibility(model_vis, modelpartition[0])
    gt = solve_gaintable(evis, model_vis, gt=gt, niter=niter, phase_only=True, gain=0.5, tol=1e-4, **kwargs)
    gt.data['gain'][...] = gain * gt.data['gain'][...] + (1 - gain) * previous_gt.data['gain'][...]
    gt.data['gain'][...] /= numpy.abs(previous_gt.data['gain'][...])
    return gt


def modelpartition_expectation_step(vis: BlockVisibility, evis_all: BlockVisibility, modelpartition, **kwargs):
    """Calculates E step in equation A12

    This is the data model for this window plus the difference between observed data and summed data models

    :param evis_all: Sum data models
    :param csm: csm element being fit
    :param kwargs:
    :return: Data model (i.e. visibility) for this csm
    """
    evis = copy_visibility(evis_all)
    tvis = copy_visibility(vis, zero=True)
    tvis = predict_skymodel_visibility(tvis, modelpartition[0], **kwargs)
    tvis = apply_gaintable(tvis, modelpartition[1])
    evis.data['vis'][...] = tvis.data['vis'][...] + vis.data['vis'][...] - evis_all.data['vis'][...]
    return evis


def modelpartition_expectation_all(vis: BlockVisibility, modelpartitions, **kwargs):
    """Calculates E step in equation A12

    This is the sum of the data models over all skymodel

    :param vis: Visibility
    :param csm: List of (skymodel, gaintable) tuples
    :param kwargs:
    :return: Sum of data models (i.e. a visibility)
    """
    evis = copy_visibility(vis, zero=True)
    tvis = copy_visibility(vis, zero=True)
    for csm in modelpartitions:
        tvis.data['vis'][...] = 0.0
        tvis = predict_skymodel_visibility(tvis, csm[0], **kwargs)
        tvis = apply_gaintable(tvis, csm[1])
        evis.data['vis'][...] += tvis.data['vis'][...]
    return evis


def modelpartition_maximisation_step(evis: BlockVisibility, modelpartition, **kwargs):
    """Calculates M step in equation A13

    This maximises the likelihood of the ssm parameters given the existing data model. Note that the skymodel and
    gaintable are done separately rather than jointly.

    :param ssm:
    :param kwargs:
    :return:
    """
    return (modelpartition_fit_skymodel(evis, modelpartition, **kwargs),
            modelpartition_fit_gaintable(evis, modelpartition, **kwargs))


def modelpartition_solve(vis, skymodels, niter=10, tol=1e-8, gain=0.25, **kwargs):
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
        evis_all = modelpartition_expectation_all(vis, model_partition)
        log.debug("modelpartition_solve: Iteration %d" % (iter))
        for window_index, csm in enumerate(model_partition):
            evis = modelpartition_expectation_step(vis, evis_all, csm, gain=gain, **kwargs)
            new_csm = modelpartition_maximisation_step(evis, csm, **kwargs)
            new_modelpartitions.append((new_csm[0], new_csm[1]))
            
            flux = new_csm[0].components[0].flux[0, 0]
            qa = qa_gaintable(new_csm[1])
            residual = qa.data['residual']
            rms_phase = qa.data['rms-phase']
            log.debug("modelpartition_solve:\t Window %d, flux %s, residual %.3f, rms phase %.3f" % (window_index,
                                                                                                  str(flux), residual,
                                                                                                  rms_phase))
        
        model_partition = [(copy_skymodel(csm[0]), copy_gaintable(csm[1])) for csm in new_modelpartitions]
    
    residual_vis = copy_visibility(vis)
    final_vis = modelpartition_expectation_all(vis, model_partition)
    residual_vis.data['vis'][...] = vis.data['vis'][...] - final_vis.data['vis'][...]
    return model_partition, residual_vis
