""" Radio interferometric calibration using the modelpartition algorithm.
"""

import logging

import numpy

from data_models.memory_data_models import BlockVisibility
from wrappers.arlexecute.calibration.calibration import solve_gaintable
from wrappers.arlexecute.calibration.operations import copy_gaintable, apply_gaintable, \
    create_gaintable_from_blockvisibility
from wrappers.arlexecute.skymodel.operations import copy_skymodel
from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility
from wrappers.arlexecute.visibility.base import copy_visibility
from workflows.arlexecute.skymodel.skymodel_arlexecute import predict_skymodel_visibility_workflow
from wrappers.arlexecute.skymodel.operations import solve_skymodel
from workflows.arlexecute.imaging.imaging_arlexecute import sum_predict_results
from wrappers.arlexecute.execution_support.arlexecute import arlexecute

log = logging.getLogger(__name__)

def create_modelpartition_list_arlexecute_workflow(vislist, skymodel_list, **kwargs):
    """Create the model partition

    Create the data model for each window, from the visibility and the skymodel

    :param comps:
    :param gt:
    :return:
    """
    
    def create_modelpartition(vis, skymodel):
        gt = create_gaintable_from_blockvisibility(vis, **kwargs)
        return (copy_skymodel(skymodel), copy_gaintable(gt))
    
    return [arlexecute.execute(create_modelpartition, nout=2)(vislist, sm) for sm in skymodel_list]


def solve_modelpartition_list_arlexecute_workflow(vislist, skymodel_list, niter=10, tol=1e-8, gain=0.25, **kwargs):
    """ Solve using modelpartition, dask.delayed wrapper

    Solve by iterating, performing E step and M step.

    :param vis: Initial visibility
    :param components: Initial components to be used
    :param gaintables: Initial gain tables to be used
    :param kwargs:
    :return: A dask graph to calculate the individual data models and the residual visibility
    """
    modelpartition_list = create_modelpartition_list_arlexecute_workflow(vislist, skymodel_list=skymodel_list, **kwargs)
    
    for iter in range(niter):
        evis_all_list = modelpartition_list_expectation_all_arlexecute_workflow(vislist, modelpartition_list)
        evislist = modelpartition_list_expectation_step_arlexecute_workflow(vislist, evis_all_list, modelpartition_list, gain=gain,
                                                                            **kwargs)
        new_modelpartition_list = modelpartition_list_maximisation_step_arlexecute_workflow(evislist, modelpartition_list, **kwargs)
        modelpartition_list = new_modelpartition_list
    
    final_vislist = modelpartition_list_expectation_all_arlexecute_workflow(vislist, modelpartition_list)
    
    def res_vis(vis, final_vis):
        residual_vis = copy_visibility(vis)
        residual_vis.data['vis'][...] = vis.data['vis'][...] - final_vis.data['vis'][...]
        return residual_vis
    
    return arlexecute.execute((modelpartition_list, arlexecute.execute(res_vis)(vislist, final_vislist)))


def modelpartition_list_expectation_step_arlexecute_workflow(vislist, evis_all_list, modelpartition_list, **kwargs):
    """Calculates E step in equation A12

    This is the data model for this window plus the difference between observed data and summed data models

    :param evis_all: Sum data models
    :param skymodel: skymodel element being fit
    :param kwargs:
    :return: Data model (i.e. visibility) for this skymodel
    """
    
    def make_e(vis, modelpartition, evis_all):
        # Return the estep for a given skymodel
        evis = copy_visibility(vis)
        tvis = copy_visibility(vis, zero=True)
        tvis = predict_skymodel_visibility_workflow(tvis, modelpartition[0])
        tvis = apply_gaintable(tvis, modelpartition[1])
        # E step is the data model for a window plus the difference between the observed data_models
        # and the summed data models or, put another way, its the observed data minus the
        # summed visibility for all other windows
        evis.data['vis'][...] = tvis.data['vis'][...] + vis.data['vis'][...] - evis_all.data['vis'][...]
        return evis
    
    return [arlexecute.execute(make_e)(vislist, csm, evis_all_list) for csm in modelpartition_list]


def modelpartition_list_expectation_all_arlexecute_workflow(vislist, modelpartition_list):
    """Calculates E step in equation A12

    This is the sum of the data models over all skymodel, It is a global sync point for modelpartition

    :param vislist: Visibility list
    :param modelpartition_list: list of modelpartition
    :return: Sum of data models (i.e. a single BlockVisibility)
    """
    
    def predict_and_apply(ovis, modelpartition):
        tvis = copy_visibility(ovis, zero=True)
        tvis = predict_skymodel_visibility_workflow(tvis, modelpartition[0])
        tvis = apply_gaintable(tvis, modelpartition[1])
        return tvis
    
    evislist = [arlexecute.execute(predict_and_apply)(vislist, csm) for csm in modelpartition_list]
    
    return arlexecute.execute(sum_predict_results, nout=1)(evislist)


def modelpartition_list_maximisation_step_arlexecute_workflow(evislist, skymodel_list, **kwargs):
    """Calculates M step in equation A13

    This maximises the likelihood of the skymodel parameters given the existing data model. Note that these are done
    separately rather than jointly.

    :param skymodel:
    :param kwargs:
    :return:
    """
    
    def make_skymodel(ev, skymodel):
        return (modelpartition_list_fit_skymodel(ev, skymodel, **kwargs),
                modelpartition_list_fit_gaintable(ev, skymodel, **kwargs))
    
    return [arlexecute.execute(make_skymodel)(evislist[i], skymodel_list[i]) for i, _ in enumerate(evislist)]


def modelpartition_list_fit_skymodel(vis, modelpartition, gain=0.1, **kwargs):
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


def modelpartition_list_fit_gaintable(evis, modelpartition, gain=0.1, niter=3, tol=1e-3, **kwargs):
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
    model_vis = predict_skymodel_visibility_workflow(model_vis, modelpartition[0])
    gt = solve_gaintable(evis, model_vis, gt=gt, niter=niter, phase_only=True, gain=0.5, tol=1e-4, **kwargs)
    gt.data['gain'][...] = gain * gt.data['gain'][...] + (1 - gain) * previous_gt.data['gain'][...]
    gt.data['gain'][...] /= numpy.abs(previous_gt.data['gain'][...])
    return gt


def modelpartition_list_expectation_step(vis: BlockVisibility, evis_all: BlockVisibility, modelpartition, **kwargs):
    """Calculates E step in equation A12

    This is the data model for this window plus the difference between observed data and summed data models

    :param evis_all: Sum data models
    :param csm: csm element being fit
    :param kwargs:
    :return: Data model (i.e. visibility) for this csm
    """
    evis = copy_visibility(evis_all)
    tvis = copy_visibility(vis, zero=True)
    tvis = predict_skymodel_visibility_workflow(tvis, modelpartition[0], **kwargs)
    tvis = apply_gaintable(tvis, modelpartition[1])
    evis.data['vis'][...] = tvis.data['vis'][...] + vis.data['vis'][...] - evis_all.data['vis'][...]
    return evis


def modelpartition_list_expectation_all(vis: BlockVisibility, modelpartitions, **kwargs):
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
        tvis = predict_skymodel_visibility_workflow(tvis, csm[0], **kwargs)
        tvis = apply_gaintable(tvis, csm[1])
        evis.data['vis'][...] += tvis.data['vis'][...]
    return evis


def modelpartition_list_maximisation_step(evis: BlockVisibility, modelpartition, **kwargs):
    """Calculates M step in equation A13

    This maximises the likelihood of the ssm parameters given the existing data model. Note that the skymodel and
    gaintable are done separately rather than jointly.

    :param ssm:
    :param kwargs:
    :return:
    """
    return (modelpartition_list_fit_skymodel(evis, modelpartition, **kwargs),
            modelpartition_list_fit_gaintable(evis, modelpartition, **kwargs))
