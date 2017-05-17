""" Common functions converted to Dask.delayed graphs. `Dask <http://dask.pydata.org/>`_ is a python-based flexible
parallel computing library for analytic computing. Dask.delayed can be used to wrap functions for deferred execution
thus allowing construction of graphs. For example, to build a graph for a major/minor cycle algorithm::

    model_graph = delayed(create_image_from_visibility)(vt, npixel=512, cellsize=0.001, npol=1)
    solution_graph = create_solve_image_graph(vt, model_graph=model_graph, psf_graph=psf_graph,
                                            invert_residual=invert_timeslice_single,
                                            predict_residual=predict_timeslice_single,
                                            iterator=vis_timeslice_iter, algorithm='hogbom',
                                            niter=1000, fractional_threshold=0.1,
                                            threshold=1.0, nmajor=3, gain=0.1)
    solution_graph.visualize()

The visualize step produces the following graph:

.. image:: ./deconvolution_dask.png
    :align: center
    :width: 1024px

The graph is executed as follows::

    solution_graph.compute()

As well as the specific graphs constructed by functions in this module, there are generic versions in the module
:mod:`arl.pipelines.generic_dask_graphs`.
"""

import collections

from dask import delayed

from arl.calibration.calibration import calibrate_visibility
from arl.calibration.operations import apply_gaintable
from arl.calibration.solvers import solve_gaintable
from arl.data.data_models import BlockVisibility, Visibility
from arl.data.parameters import get_parameter
from arl.fourier_transforms.ftprocessor import invert_2d, predict_skycomponent_blockvisibility, \
    invert_timeslice_single, predict_timeslice_single, normalize_sumwt, residual_image
from arl.image.deconvolution import deconvolve_cube, restore_cube
from arl.visibility.iterators import vis_timeslice_iter
from arl.visibility.operations import create_visibility_from_rows, \
    copy_visibility, create_visibility_from_rows, create_visibility

import logging

log = logging.getLogger(__name__)


def create_invert_graph(vis, model_graph, dopsf=False, normalize=True, invert_single=invert_2d,
                        iterator=vis_timeslice_iter, **kwargs):
    """ Create a graph to perform an invert

    The graph is constructed by iteration across the visibility

    :param vis: Visibility or BlockVisibility
    :param model_graph: Model graph
    :param dopsf: Make a PSF?
    :param normalize: Normalise the inversion?
    :param invert_single: Function to do a single invert
    :param iterator: visibility iterator
    :returns: Graph in dask.delayed format
    """
    
    def accumulate_results(results, normalize=normalize):
        log.debug('invert_graph: accumulating results')
        acc = []
        sumwt = 0.0
        for i, result in enumerate(results):
            if i > 0:
                acc.data += result[0].data
                sumwt += result[1]
            else:
                acc = result[0]
                sumwt = result[1]
        
        if normalize:
            acc = normalize_sumwt(acc, sumwt)
        
        return acc, sumwt
    
    results = list()
    
    for rows in iterator(vis, **kwargs):
        v = copy_visibility(create_visibility_from_rows(vis, rows))
        result = delayed(invert_single, pure=True)(v, model_graph, dopsf=dopsf, normalize=False, **kwargs)
        results.append(result)
    
    return delayed(accumulate_results, pure=True)(results, normalize)


def create_deconvolve_graph(dirty_graph, psf_graph, model_graph, **kwargs):
    """ Create a graph to perform a deconvolution

    :param dirty_graph: Dirty image graph
    :param psf_graph: PSF graph
    :param model_graph: Model graph
    :param kwargs: Other arguments for deconvolve_cube
    :returns: model graph
    """
    
    def deconvolve_model_only(dirty, psf, model, **kwargs):
        log.debug('deconvolve_graph: initiating deconvolution')
        result = deconvolve_cube(dirty, psf, **kwargs)[0]
        result.data += model.data
        log.debug('deconvolve_graph: finished')
        return result
    
    return delayed(deconvolve_model_only, pure=True)(dirty_graph[0], psf_graph[0], model_graph, **kwargs)

def create_apply_calibration(vis, gt_graph, **kwargs):
    """ Apply gaintables to visibility
    
    :param vis:
    :param gt_graph:
    :param kwargs:
    :return:
    """
    def apply_calibration(vis, gt):
        return apply_gaintable(vis, gt)
    
    return delayed(apply_calibration)(vis, gt_graph, **kwargs)

def create_residual_graph(vis: Visibility, model_graph, iterator=vis_timeslice_iter, **kwargs):
    """ Create a graph to calculate residual visibility

    :param vt: Visibility
    :param model_graph: Model graph
    :param iterator: visibility iterator
    :param kwargs: Other arguments for ftprocessor.residual
    :returns: residual visibility graph
    """

    def accumulate_results(results, rowses):
        log.debug('residual_graph: accumulating results')
        
        acc = []
        sumwt = 0.0
        
        for i, result in enumerate(results):
            if i > 0:
                acc.data += result[1].data
                sumwt += result[2]
            else:
                acc = result[1]
                sumwt = result[2]
        
        acc = normalize_sumwt(acc, sumwt)
        return acc, sumwt
    
    results = list()
    rowses = list()
    
    for rows in iterator(vis, **kwargs):
        rowses.append(rows)
        visslice = copy_visibility(create_visibility_from_rows(vis, rows))
        # Each result is tuple: resid vis, resid image, sumwt
        result = delayed(residual_image, pure=True)(visslice, model_graph, normalize=False, **kwargs)
        results.append(result)
    
    # We return a tuple: resid vis, residual image, sumwt
    return delayed(accumulate_results, pure=True)(results, rowses)


def create_solve_gain_graph(vis: BlockVisibility, vispred: Visibility, **kwargs):
    """ Calibrate data. Solve and return gaintables
    
    :param vis: Measured visibility
    :param kwargs:
    :return: gaintable
    """

    assert type(vis) == BlockVisibility, "vis is not a BlockVisibility"
    assert type(vispred) == BlockVisibility, "vispred is not a BlockVisibility"

    def calibrate_single(vis: BlockVisibility, vispred: BlockVisibility, **kwargs):
        log.debug('solve_gain_graph: solving for gain')
        return solve_gaintable(vis, vispred, **kwargs)
    
    return delayed(calibrate_single, pure=True)(vis, vispred, **kwargs)


def create_predict_graph(vis, model, predict_single=predict_timeslice_single, iterator=vis_timeslice_iter,
                         **kwargs):
    """

    :param vis:
    :param model:
    :param predict_single:
    :param iterator:
    :param kwargs:
    :return:
    """

    def accumulate_results(results):
        log.debug('predict_graph: accumulating results')
        i = 0
        for rows in iterator(vis, **kwargs):
            vis.data['vis'][rows] += results[i].data['vis']
            i += 1
        return vis
    
    results = list()
    
    for rows in iterator(vis, **kwargs):
        visslice = copy_visibility(create_visibility_from_rows(vis, rows))
        result = delayed(predict_single, pure=True)(visslice, model, **kwargs)
        results.append(result)
    return delayed(accumulate_results, pure=True)(results)


def create_solve_image_graph(vis, model_graph,
                             create_residual_graph=create_residual_graph,
                             create_invert_graph=create_invert_graph,
                             create_deconvolve_graph=create_deconvolve_graph, **kwargs):
    """ Create a graph to perform major/minor cycle deconvolution
    

    :param vt: Visibility
    :param model_graph: Model or model graph
    :param residual_graph: Residual graph used for residual visibilities
    :param invert_graph: Invert graph used for PSF
    :param iterator: visibility iterator
    :param first_selfcal: First cycle to selfcal, None for none
    :param kwargs: Other arguments for ftprocessor.residual
    :returns: residual visibility graph
    """
    psf_graph = create_invert_graph(vis, model_graph, dopsf=True, **kwargs)
    
    nmajor = get_parameter(kwargs, "nmajor", 5)
    
    res_graph = create_residual_graph(vis, model_graph, **kwargs)
    model_graph = create_deconvolve_graph(res_graph, psf_graph, model_graph, **kwargs)

    log.debug('solve_image_graph: defining graph')
    for cycle in range(1, nmajor):
        res_graph = create_residual_graph(vis, model_graph, **kwargs)
        model_graph = create_deconvolve_graph(res_graph, psf_graph, model_graph, **kwargs)
    
    return delayed(model_graph)


def create_restore_graph(solution_graph, psf_graph, residual_graph, **kwargs):
    """ Create a graph to restore an image
    
    :param solution_graph: Graph to make the deconvolve model
    :param psf_graph: Graph to make the PSF
    :param residual_graph: Graph to calculate residuals
    :param kwargs:
    :return:
    """
    return delayed(restore_cube, pure=True)(solution_graph, psf_graph[0], residual_graph[0], **kwargs)


def create_continuum_imaging_graph(vis, model_graph,
                                   invert_residual=invert_timeslice_single,
                                   predict_residual=predict_timeslice_single, **kwargs):
    """ Continuum imaging in dask format.
    
    :param vis: Actual Visibility
    :param model_graph: graph for the model
    :param invert_residual: function used in ftprocessor.residual
    :param predict_residual: function used in ftprocessor.residual
    :param kwargs:
    :return:
    """
    
    psf_graph = create_invert_graph(vis, model_graph, dopsf=True, normalize=False,
                                    invert_single=invert_residual, iterator=vis_timeslice_iter,
                                    timeslice=10.0)
    
    solution_graph = create_solve_image_graph(vis, model_graph=model_graph,
                                              invert_residual=invert_residual,
                                              predict_residual=predict_residual,
                                              iterator=vis_timeslice_iter, **kwargs)
    
    residual_timeslice_graph = create_residual_graph(vis, model_graph=solution_graph,
                                                     predict_residual=predict_residual,
                                                     invert_residual=invert_residual,
                                                     iterator=vis_timeslice_iter)
    
    return delayed(create_restore_graph(solution_graph=solution_graph, psf_graph=psf_graph,
                                        residual_graph=residual_timeslice_graph, **kwargs))

