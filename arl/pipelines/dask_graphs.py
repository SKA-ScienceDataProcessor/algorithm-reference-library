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

from dask import delayed

from arl.calibration.operations import apply_gaintable
from arl.calibration.solvers import solve_gaintable
from arl.data.data_models import Visibility, BlockVisibility
from arl.data.parameters import get_parameter
from arl.fourier_transforms.ftprocessor import invert_timeslice_single, predict_timeslice_single, \
    normalize_sumwt, residual_image
from arl.image.deconvolution import deconvolve_cube, restore_cube
from arl.image.operations import copy_image
from arl.visibility.operations import copy_visibility


def create_invert_graph(vis_graph_list, template_model_graph, dopsf=True, invert_single=invert_timeslice_single,
                        normalize=True, **kwargs):
    """ Sum results from invert, weighting appropriately

    """
    
    def sum_invert_results(image_list):
        for i, arg in enumerate(image_list):
            if i == 0:
                im = copy_image(arg[0])
                im.data *= arg[1]
                sumwt = arg[1]
            else:
                im.data += arg[1] * arg[0].data
                sumwt += arg[1]
        
        im = normalize_sumwt(im, sumwt)
        return im, sumwt
    
    name = 'dirty'
    if dopsf:
        name = 'psf'
    image_graph_list = list()
    for vis_graph in vis_graph_list:
        image_graph_list.append(delayed(invert_single, pure=True, name=name,
                                        nout=2)(vis_graph, template_model_graph,
                                                dopsf=dopsf, normalize=normalize,
                                                **kwargs))
    
    return delayed(sum_invert_results)(image_graph_list)


def create_deconvolve_graph(dirty_graph, psf_graph, model_graph, **kwargs):
    """Create a graph for deconvolution, adding to the model
    
    :param dirty_graph:
    :param psf_graph:
    :param model_graph:
    :param kwargs:
    :return:
    """
    
    def deconvolve(dirty, psf, model, **kwargs):
        result = deconvolve_cube(dirty, psf, **kwargs)
        result[0].data += model.data
        return result[0]
    
    return delayed(deconvolve, pure=True, nout=2)(dirty_graph[0], psf_graph[0], model_graph, **kwargs)


def create_residual_graph(vis_graph_list, model_graph, **kwargs):
    """ Sum results from invert, weighting appropriately

    """
    
    def sum_residual(image_list):
        for i, arg in enumerate(image_list):
            if i == 0:
                im = copy_image(arg[1])
                im.data *= arg[2]
                sumwt = arg[2]
            else:
                im.data += arg[2] * arg[1].data
                sumwt += arg[2]
        
        im = normalize_sumwt(im, sumwt)
        return im, sumwt
    
    image_graph_list = list()
    for vis_graph in vis_graph_list:
        image_graph_list.append(delayed(residual_image, pure=True,
                                        nout=3, name='residual')(vis_graph, model_graph, **kwargs))
    
    return delayed(sum_residual)(image_graph_list)


def create_restore_graph(vis_graph_list, model_graph, **kwargs):
    """Create graph for continuum imaging pipeline

    :param vis_graph_list:
    :param model_graph:
    :param kwargs:
    :return:
    """
    
    psf_graph = create_invert_graph(vis_graph_list, model_graph, dopsf=True, **kwargs)
    residual_graph = create_residual_graph(vis_graph_list, model_graph, **kwargs)
    return delayed(restore_cube, pure=True, nout=1)(model_graph, psf_graph[0], residual_graph[0],
                                                    **kwargs)


def create_solve_gain_graph(vis: BlockVisibility, vispred: Visibility, **kwargs):
    """ Calibrate data. Solve and return gaintables

    :param vis: Measured visibility
    :param kwargs:
    :return: gaintable
    """
    
    assert type(vis) == BlockVisibility, "vis is not a BlockVisibility"
    assert type(vispred) == BlockVisibility, "vispred is not a BlockVisibility"
    
    def calibrate_single(vis: BlockVisibility, vispred: BlockVisibility, **kwargs):
        return solve_gaintable(vis, vispred, **kwargs)
    
    return delayed(calibrate_single, pure=True)(vis, vispred, **kwargs)


def create_selfcal_graph_list(vis_graph_list, model_graph, predict_single=predict_timeslice_single,
                              **kwargs):
    """ Create a set of graphs for selfcalibration of a list of visibilities
    
    :param vis_graph_list:
    :param model_graph:
    :param predict_single:
    :param kwargs:
    :return:
    """
    
    def selfcal_single(vis_graph, model_graph, **kwargs):
        predicted = copy_visibility(vis_graph)
        predicted = predict_single(predicted, model_graph, **kwargs)
        gtsol = solve_gaintable(vis_graph, predicted, **kwargs)
        vis_graph = apply_gaintable(vis_graph, gtsol, inverse=True, **kwargs)
        return vis_graph
    
    return [delayed(selfcal_single, pure=True, nout=1)(v, model_graph, **kwargs) for v in vis_graph_list]


def create_continuum_imaging_graph(vis_graph_list, model_graph, **kwargs):
    """Create graph for continuum imaging pipeline

    :param vis_graph_list:
    :param model_graph:
    :param kwargs:
    :return:
    """
    psf_graph = create_invert_graph(vis_graph_list, model_graph, dopsf=True, **kwargs)
    residual_graph = create_residual_graph(vis_graph_list, model_graph, **kwargs)
    deconvolve_model_graph = create_deconvolve_graph(residual_graph, psf_graph, model_graph, **kwargs)
    
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(1, nmajor):
            residual_graph = create_residual_graph(vis_graph_list, deconvolve_model_graph, **kwargs)
            deconvolve_model_graph = create_deconvolve_graph(residual_graph, psf_graph,
                                                             deconvolve_model_graph, **kwargs)
    
    residual_graph = create_residual_graph(vis_graph_list, deconvolve_model_graph, **kwargs)
    restore_graph = delayed(restore_cube, pure=True, nout=1)(deconvolve_model_graph, psf_graph[0], residual_graph[0],
                                                             **kwargs)
    return delayed((deconvolve_model_graph, residual_graph, restore_graph))

def create_ical_graph(vis_graph_list, model_graph, **kwargs):
    """Create graph for ICAL pipeline
    
    :param vis_graph_list:
    :param model_graph:
    :param kwargs:
    :return:
    """
    first_selfcal = get_parameter(kwargs, "first_selfcal", 2)
    
    psf_graph = create_invert_graph(vis_graph_list, model_graph, dopsf=True, **kwargs)
    if first_selfcal == 0:
        selfcal_vis_graph_list = create_selfcal_graph_list(vis_graph_list, model_graph)
        residual_graph = create_residual_graph(selfcal_vis_graph_list, model_graph)
    else:
        residual_graph = create_residual_graph(vis_graph_list, model_graph)
    
    deconvolve_model_graph = create_deconvolve_graph(residual_graph, psf_graph, model_graph)
    
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):
            
            if cycle >= first_selfcal:
                selfcal_vis_graph_list = create_selfcal_graph_list(vis_graph_list, deconvolve_model_graph, **kwargs)
                residual_graph = create_residual_graph(selfcal_vis_graph_list, deconvolve_model_graph, **kwargs)
            
            else:
                residual_graph = create_residual_graph(vis_graph_list, deconvolve_model_graph, **kwargs)
            
            deconvolve_model_graph = create_deconvolve_graph(residual_graph, psf_graph, deconvolve_model_graph,
                                                             **kwargs)
    
    residual_graph = create_residual_graph(vis_graph_list, deconvolve_model_graph, **kwargs)
    restore_graph = delayed(restore_cube, pure=True, nout=1)(deconvolve_model_graph, psf_graph[0], residual_graph[0],
                                                             **kwargs)
    return delayed((deconvolve_model_graph, residual_graph, restore_graph))


def create_predict_graph(vis_graph_list, model_graph, predict_single=predict_timeslice_single,
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
        i = 0
        for result in results:
            vis_graph_list[i].data['vis'] += result.data['vis']
            i += 1
        return vis_graph_list
    
    results = list()
    
    for vis_graph in vis_graph_list:
        results.append(delayed(predict_single, pure=True,
                               nout=1, name='predict')(vis_graph, model_graph, **kwargs))
    
    return delayed(accumulate_results)(results)
