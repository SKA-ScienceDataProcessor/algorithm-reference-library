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

import numpy
from dask import delayed

from arl.calibration.operations import apply_gaintable
from arl.calibration.solvers import solve_gaintable
from arl.data.data_models import Visibility, BlockVisibility, Image
from arl.data.parameters import get_parameter
from arl.fourier_transforms.ftprocessor import invert_timeslice_single, predict_timeslice_single, \
    invert_wstack_single, predict_wstack_single, normalize_sumwt, residual_image
from arl.image.deconvolution import deconvolve_cube, restore_cube
from arl.image.gather_scatter import image_scatter, image_gather
from arl.image.operations import copy_image, create_empty_image_like
from arl.visibility.gather_scatter import visibility_scatter_w, visibility_gather_w
from arl.visibility.operations import copy_visibility


def create_zero_vis_graph_list(vis_graph_list, **kwargs):
    """ Initialise vis to zero: creates new data holders

    :param vis_graph_list:
    :param kwargs:
    :return: List of vis_graphs
   """
    
    def zerovis(vis):
        if vis is not None:
            zerovis = copy_visibility(vis)
            zerovis.data['vis'][...] = 0.0
            return zerovis
        else:
            return None
    
    return [delayed(zerovis, pure=True, nout=1)(v) for v in vis_graph_list]


def create_subtract_vis_graph_list(vis_graph_list, model_vis_graph_list, **kwargs):
    """ Initialise vis to zero

    :param vis_graph_list:
    :param kwargs:
    :return: List of vis_graphs
   """
    
    def subtract_vis(vis, model_vis):
        if vis is not None and model_vis is not None:
            assert vis.vis.shape == model_vis.vis.shape
            subvis = copy_visibility(vis)
            subvis.data['vis'][...] -= model_vis.data['vis'][...]
            return subvis
        else:
            return None
    
    return [delayed(subtract_vis, pure=True, nout=1)(vis=vis_graph_list[i],
                                                     model_vis=model_vis_graph_list[i])
            for i in range(len(vis_graph_list))]


def create_invert_graph(vis_graph_list, template_model_graph, dopsf=False,
                        invert_single=invert_timeslice_single,
                        normalize=True, **kwargs):
    """ Sum results from invert, weighting appropriately

    :param vis_graph_list:
    :param model_graph:
    :param invert_single: Invert for a single Visibility
    :param kwargs:
    :returns: Graph for invert
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
    
    image_graph_list = list()
    for vis_graph in vis_graph_list:
        if vis_graph is not None:
            image_graph_list.append(delayed(invert_single, pure=True, nout=2)(vis_graph, template_model_graph,
                                                                              dopsf=dopsf, normalize=normalize,
                                                                              **kwargs))
    
    return delayed(sum_invert_results)(image_graph_list)


def create_invert_wstack_graph(vis_graph_list, template_model_graph, dopsf=False,
                               vis_slices=1, normalize=True, **kwargs):
    """ Sum results from invert_wstack, weighting appropriately

    :param vis_graph_list:
    :param model_graph:
    :param vis_slices: Number of visibility slices in w
    :param kwargs:
    :returns: Graph for invert
    """
    
    def sum_invert_results(image_list):
        first = True
        for i, arg in enumerate(image_list):
            if arg is not None:
                if first:
                    im = copy_image(arg[0])
                    im.data *= arg[1]
                    sumwt = arg[1]
                    first = False
                else:
                    im.data += arg[1] * arg[0].data
                    sumwt += arg[1]
        assert numpy.sum(sumwt) > 0.0
        im = normalize_sumwt(im, sumwt)
        return im, sumwt
    
    def invert_ignore_None(vis, *args, **kwargs):
        if vis is not None:
            return invert_wstack_single(vis, *args, **kwargs)
        else:
            return None
    
    image_graph_list = list()
    for vis_graph in vis_graph_list:
        if vis_graph is not None:
            scatter_vis_graphs = delayed(visibility_scatter_w, nout=vis_slices)(vis_graph, vis_slices=vis_slices,
                                                                                **kwargs)
            for scatter_vis_graph in scatter_vis_graphs:
                image_graph_list.append(delayed(invert_ignore_None, pure=True, nout=2)(scatter_vis_graph,
                                                                                       template_model_graph,
                                                                                       dopsf=dopsf,
                                                                                       normalize=normalize,
                                                                                       **kwargs))
    
    return delayed(sum_invert_results)(image_graph_list)


def create_invert_facet_graph(vis_graph_list, template_model_graph, dopsf=False,
                              invert_single=invert_timeslice_single,
                              facets=1, normalize=True, **kwargs):
    """ Sum results from invert, weighting appropriately, allows faceting

    :param vis_graph_list:
    :param model_graph:
    :param invert_single: Invert for a single Visibility
    :param kwargs:
    :returns: Graph for invert
   """
    
    def gather(results, template_model, facets, **kwargs):
        # Results contains the images for each facet, after adding across vis_graphs
        image_results = create_empty_image_like(template_model)
        image_results = image_gather([result[0] for result in results], image_results,
                                     facets=facets)
        # For the gather, assume all are the same weight
        sumwt = results[0][1]
        
        return image_results, sumwt
    
    # Scatter the model in facets
    model_graphs = delayed(image_scatter, nout=facets ** 2, pure=True)(template_model_graph,
                                                                       facets=facets)
    
    # For each facet, invert over the vis_graph
    results = [create_invert_graph(vis_graph_list, model_graph, dopsf=dopsf,
                                   invert_single=invert_single, normalize=normalize, **kwargs)
               for model_graph in model_graphs]
    # Now we have a list containing the facet images added over vis_graph. We can now
    # gather those images into one image
    return delayed(gather, nout=2, pure=True)(results, template_model_graph, facets=facets, **kwargs)


def create_predict_graph(vis_graph_list, model_graph, predict_single=predict_timeslice_single,
                         **kwargs):
    """

    :param vis_graph_list:
    :param model_graph:
    :param predict_single: Predict function to be used (predict_time_slice_single)
    :param kwargs:
    :return: List of vis_graphs
   """
    
    def predict_and_sum(vis, model, **kwargs):
        if vis is not None:
            predicted = copy_visibility(vis)
            predicted = predict_single(predicted, model, predict_single=predict_single, **kwargs)
            predicted.data['vis'] = vis.data['vis'] + predicted.data['vis']
            return predicted
        else:
            return None
    
    return [delayed(predict_and_sum, pure=True, nout=1)(v, model_graph, **kwargs) for v in vis_graph_list]


def create_predict_wstack_graph(vis_graph_list, model_graph, predict_single=predict_wstack_single,
                                vis_slices=1, **kwargs):
    """

    :param vis_graph_list:
    :param model_graph:
    :param predict_single: Predict function to be used (predict_time_slice_single)
    :param kwargs:
    :return: List of vis_graphs
   """
    
    def predict_and_sum(vis, model, **kwargs):
        if vis is not None:
            predicted = copy_visibility(vis)
            predicted = predict_single(predicted, model, predict_single=predict_single, **kwargs)
            predicted.data['vis'] = vis.data['vis'] + predicted.data['vis']
            return predicted
        else:
            return None
    
    def predict_ignore_None(vis, *args, **kwargs):
        if vis is not None:
            return predict_wstack_single(vis, *args, **kwargs)
        else:
            return None
    
    predicted_vis_list = list()
    for vis_graph in vis_graph_list:
        predict_list = list()
        scatter_vis_graphs = delayed(visibility_scatter_w, nout=vis_slices)(vis_graph, vis_slices=vis_slices, **kwargs)
        for scatter_vis_graph in scatter_vis_graphs:
            predict_list.append(delayed(predict_and_sum, pure=True, nout=1)(scatter_vis_graph, model_graph, **kwargs))
        predicted_vis_list.append(delayed(visibility_gather_w, nout=1)(predict_list, vis_graph, vis_slices=vis_slices,
                                                                       **kwargs))
    
    return predicted_vis_list


def create_predict_facet_graph(vis_graph_list, model_graph, predict_single=predict_wstack_single,
                               facets=2, **kwargs):
    """ Predict visibility from a model using facets

    :param vis_graph_list:
    :param model_graph:
    :param predict_single: Predict function to be used (predict_time_slice_single)
    :param facets: Number of facets on axis (4)
    :param kwargs:
    :return: List of vis_graphs
    """
    
    def predict_and_accumulate(vis, model, **kwargs):
        if vis is not None:
            predicted = copy_visibility(vis)
            predicted = predict_single(predicted, model, predict_single=predict_single, **kwargs)
            vis.data['vis'] += predicted.data['vis']
            return vis
        else:
            return None
    
    facet_model_graphs = delayed(image_scatter, nout=facets ** 2, pure=True)(model_graph,
                                                                             facets=facets)
    # There is a dependency issue here so we chain the predicts
    accumulate_vis_graphs = list()
    for vis_graph in vis_graph_list:
        if vis_graph is not None:
            for ifacet, facet_model_graph in enumerate(facet_model_graphs):
                if ifacet == 0:
                    accumulate_vis_graph = delayed(predict_and_accumulate, pure=True, nout=1)(vis_graph,
                                                                                              facet_model_graph,
                                                                                              **kwargs)
                else:
                    accumulate_vis_graph = delayed(predict_and_accumulate, pure=True, nout=1)(accumulate_vis_graph,
                                                                                              facet_model_graph,
                                                                                              **kwargs)
        accumulate_vis_graphs.append(accumulate_vis_graph)
    return accumulate_vis_graphs


def create_residual_graph(vis_graph_list, model_graph,
                          invert_residual=invert_timeslice_single,
                          predict_residual=predict_timeslice_single,
                          **kwargs):
    """ Create a graph to calculate residual image
    
    :param vis_graph_list:
    :param model_graph:
    :param c_invert_facet_graph:
    :param c_predict_facet_graph:
    :param predict_single:
    :param invert_single:
    :param kwargs:
    :return: graph
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
        if vis_graph_list is not None:
            image_graph_list.append(delayed(residual_image, pure=True,
                                            nout=3, name='residual')(vis_graph, model_graph,
                                                                    invert_residual=invert_residual,
                                                                    predict_residual=predict_residual,
                                                                    **kwargs))
    
    return delayed(sum_residual)(image_graph_list)


def create_residual_graph(vis_graph_list, model_graph, facets=4,
                                c_invert_graph=create_invert_graph,
                                c_predict_graph=create_predict_graph,
                                predict_single=predict_timeslice_single,
                                invert_single=invert_timeslice_single,
                                **kwargs):
    """ Create a graph to calculate residual image using facets

    :param vis_graph_list:
    :param model_graph:
    :param facets: Number of facets on each axis
    :param c_invert_graph:
    :param c_predict_graph:
    :param predict_single:
    :param invert_single:
    :param kwargs:
    :return:
    """
    model_vis_graph_list = create_zero_vis_graph_list(vis_graph_list)
    model_vis_graph_list = c_predict_graph(model_vis_graph_list, model_graph, facets=facets,
                                                 predict_single=predict_single,
                                                 **kwargs)
    residual_vis_graph_list = create_subtract_vis_graph_list(vis_graph_list, model_vis_graph_list)
    return c_invert_graph(residual_vis_graph_list, model_graph, dopsf=False, normalize=True,
                                facets=facets, invert_single=invert_single, **kwargs)


def create_residual_wstack_graph(vis_graph_list, model_graph, vis_slices=1,
                                 c_invert_graph=create_invert_wstack_graph,
                                 c_predict_graph=create_predict_wstack_graph,
                                 **kwargs):
    """ Create a graph to calculate residual image using facets

    :param vis_graph_list:
    :param model_graph:
    :param vis_slices: Number of fw planes
    :param c_invert_facet_graph:
    :param c_predict_facet_graph:
    :param kwargs:
    :return:
    """
    model_vis_graph_list = create_zero_vis_graph_list(vis_graph_list)
    model_vis_graph_list = c_predict_graph(model_vis_graph_list, model_graph, vis_slices=vis_slices,
                                           **kwargs)
    residual_vis_graph_list = create_subtract_vis_graph_list(vis_graph_list, model_vis_graph_list)
    return c_invert_graph(residual_vis_graph_list, model_graph, dopsf=False, normalize=True,
                          vis_slices=vis_slices, **kwargs)


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


def create_deconvolve_facet_graph(dirty_graph, psf_graph, model_graph,
                                  facets=2, **kwargs):
    """Create a graph for deconvolution, adding to the model
    
    Does deconvolution facet-by-facet. Currently does nothing very sensible about the
    edges.

    :param dirty_graph:
    :param psf_graph: Must be the size of a facet
    :param model_graph: Current model
    :param facets: Number of facets on each axis
    :param kwargs:
    :return:
    """
    
    def deconvolve(dirty, psf, **kwargs):
        assert type(dirty) == Image
        assert type(psf) == Image
        result = deconvolve_cube(dirty, psf, **kwargs)
        return result[0]
    
    def add_model(output, model):
        assert type(output) == Image
        assert type(model) == Image
        output.data += model.data
        return output
    
    output = delayed(create_empty_image_like, nout=1, pure=True)(model_graph)
    dirty_graphs = delayed(image_scatter, nout=facets ** 2, pure=True)(dirty_graph[0], facets=facets)
    results = [delayed(deconvolve)(dirty_graph, psf_graph[0], **kwargs)
               for dirty_graph in dirty_graphs]
    result = delayed(image_gather, nout=1, pure=True)(results, output, facets=facets)
    return delayed(add_model, nout=1, pure=True)(result, model_graph)


def create_solve_gain_graph(vis: BlockVisibility, vispred: Visibility, **kwargs):
    """ Create a graph to solve for gain table

    :param vis: Measured visibility
    :param vispred: Predicted visibility
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
    
    def selfcal_single(vis, model, **kwargs):
        if vis is not None:
            predicted = copy_visibility(vis)
            predicted = predict_single(predicted, model, **kwargs)
            gtsol = solve_gaintable(vis, predicted, **kwargs)
            vis = apply_gaintable(vis, gtsol, inverse=True, **kwargs)
            return vis
        else:
            return None
    
    return [delayed(selfcal_single, pure=True, nout=1)(v, model_graph, **kwargs) for v in vis_graph_list]


def create_continuum_imaging_pipeline_graph(vis_graph_list, model_graph,
                                            c_deconvolve_graph=create_deconvolve_graph,
                                            c_invert_graph=create_invert_graph,
                                            c_residual_graph=create_residual_graph,
                                            **kwargs):
    """ Create graph for the continuum imaging pipeline using facets
    
    :param vis_graph_list:
    :param model_graph:
    :param c_deconvolve_graph:
    :param c_invert_graph:
    :param c_residual_graph:
    :param facets:
    :param kwargs:
    :return:
    """
    psf_graph = c_invert_graph(vis_graph_list, model_graph, dopsf=True, **kwargs)
    residual_graph = c_residual_graph(vis_graph_list, model_graph, **kwargs)
    deconvolve_model_graph = c_deconvolve_graph(residual_graph, psf_graph, model_graph, **kwargs)
    
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(1, nmajor):
            residual_graph = c_residual_graph(vis_graph_list, deconvolve_model_graph, **kwargs)
            deconvolve_model_graph = c_deconvolve_graph(residual_graph, psf_graph,
                                                        deconvolve_model_graph, **kwargs)
    
    residual_graph = c_residual_graph(vis_graph_list, deconvolve_model_graph, **kwargs)
    restore_graph = delayed(restore_cube, pure=True, nout=1)(deconvolve_model_graph,
                                                             psf_graph[0], residual_graph[0],
                                                             **kwargs)
    return delayed((deconvolve_model_graph, residual_graph, restore_graph))


def create_spectral_line_imaging_pipeline_graph(vis_graph_list, model_graph, continuum_model_graph=None,
                                                c_deconvolve_graph=create_deconvolve_graph,
                                                c_invert_graph=create_invert_graph,
                                                c_predict_graph=create_predict_graph,
                                                c_residual_graph=create_residual_graph,
                                                vis_slices=1, facets=2, **kwargs):
    """Create graph for spectral line imaging pipeline

    :param vis_graph_list: List of visibility graphs
    :param model_graph: Spectral line model graph
    :param continuum_model_graph: Continuum model graph
    :param kwargs:
    :return: graphs of (deconvolved model, residual, restored)
    """
    if continuum_model_graph is not None:
        vis_graph_list = c_predict_graph(vis_graph_list, continuum_model_graph, vis_slices=vis_slices, **kwargs)
    
    psf_graph = c_invert_graph(vis_graph_list, model_graph, dopsf=True, **kwargs)
    residual_graph = c_residual_graph(vis_graph_list, model_graph, **kwargs)
    deconvolve_model_graph = c_deconvolve_graph(residual_graph, psf_graph, model_graph, **kwargs)
    
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(1, nmajor):
            residual_graph = c_residual_graph(vis_graph_list, deconvolve_model_graph, **kwargs)
            deconvolve_model_graph = c_deconvolve_graph(residual_graph, psf_graph, deconvolve_model_graph, **kwargs)
    
    residual_graph = c_residual_graph(vis_graph_list, deconvolve_model_graph, **kwargs)
    restore_graph = delayed(restore_cube, pure=True, nout=1)(deconvolve_model_graph, psf_graph[0],
                                                             residual_graph[0],
                                                             **kwargs)
    return delayed((deconvolve_model_graph, residual_graph, restore_graph))


def create_ical_pipeline_graph(vis_graph_list, model_graph,
                               c_deconvolve_graph=create_deconvolve_graph,
                               c_invert_graph=create_invert_graph,
                               c_residual_graph=create_residual_graph,
                               vis_slices=1, facets=2, **kwargs):
    """Create graph for ICAL pipeline
    
    :param vis_graph_list:
    :param model_graph:
    :param kwargs:
    :return:
    """
    psf_graph = c_invert_graph(vis_graph_list, model_graph, dopsf=True, **kwargs)
    
    first_selfcal = get_parameter(kwargs, "first_selfcal", 2)
    if first_selfcal == 0:
        vis_graph_list = create_selfcal_graph_list(vis_graph_list, model_graph, **kwargs)
    residual_graph = c_residual_graph(vis_graph_list, model_graph, **kwargs)
    deconvolve_model_graph = c_deconvolve_graph(residual_graph, psf_graph, model_graph, **kwargs)
    
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):
            if cycle >= first_selfcal:
                vis_graph_list = create_selfcal_graph_list(vis_graph_list,
                                                           deconvolve_model_graph, **kwargs)
            residual_graph = c_residual_graph(vis_graph_list, deconvolve_model_graph, **kwargs)
            
            deconvolve_model_graph = c_deconvolve_graph(residual_graph, psf_graph,
                                                        deconvolve_model_graph, **kwargs)
        residual_graph = c_residual_graph(vis_graph_list, deconvolve_model_graph,
                                          vis_slices=vis_slices, facets=facets,
                                          **kwargs)
    
    restore_graph = delayed(restore_cube, pure=True, nout=1)(deconvolve_model_graph,
                                                             psf_graph[0], residual_graph[0],
                                                             **kwargs)
    return delayed((deconvolve_model_graph, residual_graph, restore_graph))
