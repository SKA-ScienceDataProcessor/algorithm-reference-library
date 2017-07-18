""" Common functions converted to Dask.delayed graphs. `Dask <http://dask.pydata.org/>`_ is a python-based flexible
parallel computing library for analytic computing. Dask.delayed can be used to wrap functions for deferred execution
thus allowing construction of graphs. For example, to build a graph for a major/minor cycle algorithm::

    model_graph = delayed(create_image_from_visibility)(vt, npixel=512, cellsize=0.001, npol=1)
    solution_graph = create_solve_image_graph(vt, model_graph=model_graph, psf_graph=psf_graph,
                                            invert_residual=invert_timeslice,
                                            predict_residual=predict_timeslice,
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

Note that all parameters here should be passed using the kwargs mechanism. The exceptions
are those needed to define the size of a graph. Since delayed graphs are not Iterable
by default, it is necessary to use the nout= parameter to delayed to specify the
graph size.
"""

from typing import List

import numpy
from dask import delayed, bag

from arl.calibration.operations import apply_gaintable
from arl.calibration.solvers import solve_gaintable
from arl.data.data_models import Image
from arl.image.deconvolution import deconvolve_cube
from arl.image.gather_scatter import image_scatter, image_gather
from arl.image.operations import copy_image, create_empty_image_like
from arl.imaging import predict_2d, invert_2d, invert_wstack_single, predict_wstack_single, normalize_sumwt
from arl.visibility.gather_scatter import visibility_scatter_w, visibility_gather_w, \
    visibility_scatter_channel, visibility_gather_channel
from arl.visibility.operations import copy_visibility


def create_zero_vis_graph_list(vis_graph_list: List[delayed], **kwargs) -> List[delayed]:
    """ Initialise vis to zero: creates new data holders

    :param vis_graph_list:
    :param kwargs: Parameters for functions in graphs
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


def create_subtract_vis_graph_list(vis_graph_list: List[delayed], model_vis_graph_list: List[delayed],
                                   **kwargs) -> List[delayed]:
    """ Initialise vis to zero

    :param vis_graph_list:
    :param kwargs: Parameters for functions in graphs
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


def create_invert_graph(vis_graph_list: List[delayed], template_model_graph: delayed, dopsf=False, invert=invert_2d,
                        normalize=True, **kwargs) -> delayed:
    """ Sum results from invert iterating over the vis_graph_list

    :param vis_graph_list:
    :param model_graph:
    :param invert: Invert for a single Visibility set
    :param kwargs: Parameters for functions in graphs
    :return: delayed for invert
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
        
        im = normalize_sumwt(im, sumwt)
        return im, sumwt
    
    def invert_ignore_None(vis, *args, **kwargs):
        if vis is not None:
            return invert(vis, *args, **kwargs)
        else:
            return None
    
    image_graph_list = list()
    for vis_graph in vis_graph_list:
        image_graph_list.append(delayed(invert_ignore_None, pure=True, nout=2)(vis_graph, template_model_graph,
                                                                               dopsf=dopsf, normalize=normalize,
                                                                               **kwargs))
    
    return delayed(sum_invert_results)(image_graph_list)


def create_invert_wstack_graph(vis_graph_list: List[delayed], template_model_graph: delayed, vis_slices,
                               dopsf=False, normalize=True, invert=invert_wstack_single, **kwargs) -> delayed:
    """ Sum invert results using wstacking, iterating over the vis_graph_list and w
    
    :param vis_graph_list:
    :param model_graph:
    :param dopsf: Make psf (False)
    :param vis_slices: Number of visibility slices in w stacking
    :param kwargs: Parameters for functions in graphs
    :return: delayed for invert
    """
    
    def sum_invert_wstack_results(image_list):
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
        if numpy.sum(sumwt) > 0.0:
            im = normalize_sumwt(im, sumwt)
        return im, sumwt
    
    def invert_wstack_ignore_None(vis, *args, **kwargs):
        if vis is not None:
            return invert(vis, *args, **kwargs)
        else:
            return None
    
    image_graph_list = list()
    for vis_graph in vis_graph_list:
        if vis_graph is not None:
            scatter_vis_graphs = delayed(visibility_scatter_w, nout=vis_slices)(vis_graph, vis_slices=vis_slices,
                                                                                **kwargs)
            for scatter_vis_graph in scatter_vis_graphs:
                image_graph_list.append(delayed(invert_wstack_ignore_None, pure=True, nout=2)(scatter_vis_graph,
                                                                                              template_model_graph,
                                                                                              dopsf=dopsf,
                                                                                              normalize=normalize,
                                                                                              **kwargs))
    
    return delayed(sum_invert_wstack_results)(image_graph_list)


def create_invert_facet_graph(vis_graph_list: List[delayed], template_model_graph: delayed, dopsf=False, normalize=True,
                              facets=1, **kwargs) -> delayed:
    """ Sum results from invert, iterating over the vis_graph_list, allows faceting

    :param vis_graph_list:
    :param model_graph:
    :param invert: Invert for a single Visibility
    :param kwargs: Parameters for functions in graphs
    :return: delayed for invert
   """
    
    def gather_invert_results(results, template_model, facets, **kwargs):
        # Results contains the images for each facet, after adding across vis_graphs
        image_results = create_empty_image_like(template_model)
        image_results = image_gather([result[0] for result in results], image_results,
                                     facets=facets)
        # For the gather, assume all are the same weight
        sumwt = results[0][1]
        
        return image_results, sumwt
    
    # Scatter the model in facets
    model_graphs = delayed(image_scatter, nout=facets ** 2, pure=True)(template_model_graph, facets=facets)
    
    # For each facet, invert over the vis_graph
    results = [create_invert_graph(vis_graph_list, model_graph, dopsf=dopsf, normalize=normalize, **kwargs)
               for model_graph in model_graphs]
    # Now we have a list containing the facet images added over vis_graph. We can now
    # gather those images into one image
    return delayed(gather_invert_results, nout=2, pure=True)(results, template_model_graph, facets=facets, **kwargs)


def create_invert_facet_wstack_graph(vis_graph_list: List[delayed], template_model_graph: delayed, dopsf=False,
                                     normalize=True, facets=2, **kwargs) -> delayed:
    """ Sum results from invert, iterating over the vis_graph_list, allows faceting

    :param vis_graph_list:
    :param model_graph:
    :param invert_single: Invert for a single Visibility
    :param kwargs: Parameters for functions in graphs
    :return: delayed for invert
   """
    
    def gather_invert_results(results, template_model, facets, **kwargs):
        # Results contains the images for each facet, after adding across vis_graphs
        image_results = create_empty_image_like(template_model)
        image_results = image_gather([result[0] for result in results], image_results,
                                     facets=facets)
        # For the gather, assume all are the same weight
        sumwt = results[0][1]
        
        return image_results, sumwt
    
    # Scatter the model in facets
    model_graphs = delayed(image_scatter, nout=facets ** 2, pure=True)(template_model_graph, facets=facets)
    
    # For each facet, invert over the vis_graph
    results = [create_invert_wstack_graph(vis_graph_list, model_graph, dopsf=dopsf, normalize=normalize, **kwargs)
               for model_graph in model_graphs]
    # Now we have a list containing the facet images added over vis_graph. We can now
    # gather those images into one image
    return delayed(gather_invert_results, nout=2, pure=True)(results, template_model_graph, facets=facets, **kwargs)


def create_predict_graph(vis_graph_list: List[delayed], model_graph: delayed, predict=predict_2d, **kwargs) -> \
        List[delayed]:
    """Predict from model_graph, iterating over the vis_graph_list

    :param vis_graph_list:
    :param model_graph:
    :param predict: Predict function to be used (predict_t2d)
    :param kwargs: Parameters for functions in graphs Parameters for functions in graphs
    :return: List of vis_graphs
   """
    
    def predict_and_sum(vis, model, **kwargs):
        if vis is not None:
            predicted = copy_visibility(vis)
            predicted = predict(predicted, model, **kwargs)
            predicted.data['vis'] = vis.data['vis'] + predicted.data['vis']
            return predicted
        else:
            return None
    
    return [delayed(predict_and_sum, pure=True, nout=1)(v, model_graph, **kwargs) for v in vis_graph_list]


def create_predict_facet_graph(vis_graph_list: List[delayed], model_graph: delayed, predict=predict_2d, facets=2,
                               **kwargs) -> List[delayed]:
    """ Predict visibility from a model using facets

    :param vis_graph_list:
    :param model_graph:
    :param predict: Predict function to be used (predict_2d)
    :param facets: Number of facets on axis (4)
    :param kwargs: Parameters for functions in graphs
    :return: List of vis_graphs
    """
    
    def predict_facets_and_accumulate(vis, model, **kwargs):
        if vis is not None:
            predicted = copy_visibility(vis)
            predicted = predict(predicted, model, **kwargs)
            vis.data['vis'] += predicted.data['vis']
            return vis
        else:
            return None
    
    # Note that we need to know the number of facets in order to define the size of facet_model_graphs
    facet_model_graphs = delayed(image_scatter, nout=facets ** 2, pure=True)(model_graph,
                                                                             facets=facets)
    # There is a dependency issue here so we chain the predicts
    accumulate_vis_graphs = list()
    for vis_graph in vis_graph_list:
        if vis_graph is not None:
            for ifacet, facet_model_graph in enumerate(facet_model_graphs):
                if ifacet == 0:
                    accumulate_vis_graph = delayed(predict_facets_and_accumulate, pure=True, nout=1)(vis_graph,
                                                                                                     facet_model_graph,
                                                                                                     **kwargs)
                else:
                    accumulate_vis_graph = delayed(predict_facets_and_accumulate, pure=True, nout=1)(
                        accumulate_vis_graph,
                        facet_model_graph,
                        **kwargs)
        accumulate_vis_graphs.append(accumulate_vis_graph)
    return accumulate_vis_graphs


def create_predict_wstack_graph(vis_graph_list: List[delayed], model_graph: delayed, vis_slices,
                                predict=predict_wstack_single, **kwargs) -> List[delayed]:
    """Predict using wstacking, iterating over the vis_graph_list and w

    :param vis_graph_list:
    :param model_graph:
    :param kwargs: Parameters for functions in graphs
    :return: List of vis_graphs
   """
    
    def predict_wstack_and_sum(vis, model, **kwargs):
        if vis is not None:
            predicted = copy_visibility(vis)
            predicted = predict(predicted, model, **kwargs)
            predicted.data['vis'] = vis.data['vis'] + predicted.data['vis']
            return predicted
        else:
            return None
    
    predicted_vis_list = list()
    for vis_graph in vis_graph_list:
        predict_list = list()
        scatter_vis_graphs = delayed(visibility_scatter_w, nout=vis_slices)(vis_graph, vis_slices=vis_slices, **kwargs)
        for scatter_vis_graph in scatter_vis_graphs:
            predict_list.append(delayed(predict_wstack_and_sum, pure=True, nout=1)(scatter_vis_graph,
                                                                                   model_graph,
                                                                                   **kwargs))
        predicted_vis_list.append(delayed(visibility_gather_w, nout=1)(predict_list, vis_graph, vis_slices=vis_slices,
                                                                       **kwargs))
    
    return predicted_vis_list


def create_predict_facet_wstack_graph(vis_graph_list: List[delayed], model_graph: delayed, vis_slices, facets,
                                      predict=predict_wstack_single, **kwargs) -> List[delayed]:
    """Predict using wstacking, iterating over the vis_graph_list and w

    :param vis_graph_list:
    :param model_graph:
    :param kwargs: Parameters for functions in graphs
    :return: List of vis_graphs
   """
    
    def predict_facets_and_accumulate(vis, model, **kwargs):
        if vis is not None:
            predicted = copy_visibility(vis)
            predicted = predict(predicted, model, **kwargs)
            vis.data['vis'] += predicted.data['vis']
            return vis
        else:
            return None
            
            # Note that we need to know the number of facets in order to define the size of facet_model_graphs
    
    facet_model_graphs = delayed(image_scatter, nout=facets ** 2, pure=True)(model_graph,
                                                                             facets=facets)
    predicted_vis_list = list()
    for vis_graph in vis_graph_list:
        scatter_vis_graphs = delayed(visibility_scatter_w, nout=vis_slices)(vis_graph, vis_slices=vis_slices,
                                                                            **kwargs)
        
        accumulate_vis_graphs = list()
        for scatter_vis_graph in scatter_vis_graphs:
            for ifacet, facet_model_graph in enumerate(facet_model_graphs):
                if ifacet == 0:
                    accumulate_vis_graph = delayed(predict_facets_and_accumulate,
                                                   pure=True, nout=1)(scatter_vis_graph, facet_model_graphs[0],
                                                                      **kwargs)
                else:
                    accumulate_vis_graph = delayed(predict_facets_and_accumulate,
                                                   pure=True, nout=1)(accumulate_vis_graph, facet_model_graph,
                                                                      **kwargs)
            accumulate_vis_graphs.append(accumulate_vis_graph)
        
        predicted_vis_list.append(delayed(visibility_gather_w, nout=1)(accumulate_vis_graphs, vis_graph,
                                                                       vis_slices=vis_slices, **kwargs))
    
    return predicted_vis_list


def create_residual_graph(vis_graph_list: List[delayed], model_graph: delayed, **kwargs) -> delayed:
    """ Create a graph to calculate residual image using facets

    :param vis_graph_list:
    :param model_graph:
    :param invert_single:
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    model_vis_graph_list = create_zero_vis_graph_list(vis_graph_list)
    model_vis_graph_list = create_predict_graph(model_vis_graph_list, model_graph, **kwargs)
    residual_vis_graph_list = create_subtract_vis_graph_list(vis_graph_list, model_vis_graph_list)
    return create_invert_graph(residual_vis_graph_list, model_graph, dopsf=False, normalize=True, **kwargs)


def create_residual_facet_graph(vis_graph_list: List[delayed], model_graph: delayed, **kwargs) -> delayed:
    """ Create a graph to calculate residual image using facets

    :param vis_graph_list:
    :param model_graph:
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    model_vis_graph_list = create_zero_vis_graph_list(vis_graph_list)
    model_vis_graph_list = create_predict_facet_graph(model_vis_graph_list, model_graph, **kwargs)
    residual_vis_graph_list = create_subtract_vis_graph_list(vis_graph_list, model_vis_graph_list)
    return create_invert_facet_graph(residual_vis_graph_list, model_graph, dopsf=False, normalize=True,
                                     **kwargs)


def create_residual_wstack_graph(vis_graph_list: List[delayed], model_graph: delayed, **kwargs) -> delayed:
    """ Create a graph to calculate residual image using w stacking

    :param vis_graph_list:
    :param model_graph:
    :param vis_slices: Number of w planes
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    model_vis_graph_list = create_zero_vis_graph_list(vis_graph_list)
    model_vis_graph_list = create_predict_wstack_graph(model_vis_graph_list, model_graph, **kwargs)
    residual_vis_graph_list = create_subtract_vis_graph_list(vis_graph_list, model_vis_graph_list)
    return create_invert_wstack_graph(residual_vis_graph_list, model_graph, dopsf=False, normalize=True,
                                      **kwargs)


def create_residual_facet_wstack_graph(vis_graph_list: List[delayed], model_graph: delayed, **kwargs) -> delayed:
    """ Create a graph to calculate residual image using w stacking and faceting

    :param vis_graph_list:
    :param model_graph:
    :param vis_slices: Number of w planes
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    model_vis_graph_list = create_zero_vis_graph_list(vis_graph_list)
    model_vis_graph_list = create_predict_facet_wstack_graph(model_vis_graph_list, model_graph, **kwargs)
    residual_vis_graph_list = create_subtract_vis_graph_list(vis_graph_list, model_vis_graph_list)
    return create_invert_facet_wstack_graph(residual_vis_graph_list, model_graph, dopsf=False, normalize=True,
                                            **kwargs)


def create_deconvolve_graph(dirty_graph: delayed, psf_graph: delayed, model_graph: delayed, **kwargs) -> delayed:
    """Create a graph for deconvolution, adding to the model

    :param dirty_graph:
    :param psf_graph:
    :param model_graph:
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    
    def deconvolve(dirty, psf, model, **kwargs):
        result = deconvolve_cube(dirty, psf, **kwargs)
        result[0].data += model.data
        return result[0]
    
    return delayed(deconvolve, pure=True, nout=2)(dirty_graph[0], psf_graph[0], model_graph, **kwargs)


def create_deconvolve_facet_graph(dirty_graph: delayed, psf_graph: delayed, model_graph: delayed, facets=1,
                                  **kwargs) -> delayed:
    """Create a graph for deconvolution by facets, adding to the model
    
    Does deconvolution facet-by-facet. Currently does nothing very sensible about the
    edges.

    :param dirty_graph:
    :param psf_graph: Must be the size of a facet
    :param model_graph: Current model
    :param facets: Number of facets on each axis
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    
    def deconvolve_facet(dirty, psf, **kwargs):
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
    results = [delayed(deconvolve_facet)(dirty_graph, psf_graph[0], **kwargs)
               for dirty_graph in dirty_graphs]
    result = delayed(image_gather, nout=1, pure=True)(results, output, facets=facets)
    return delayed(add_model, nout=1, pure=True)(result, model_graph)


def create_selfcal_graph_list(vis_graph_list: List[delayed], model_graph: delayed,
                              vis_slices, global_solution=False,
                              c_predict_vis_graph=create_predict_wstack_graph, **kwargs) -> List[delayed]:
    """ Create a set of graphs for (optionally global) selfcalibration of a list of visibilities
    
    The visibilities are gathered to a single visibility data set which is then self-calibrated. The result is
    then scattered.

    :param vis_graph_list:
    :param model_graph:
    :param vis_slices:
    :param c_predict_vis_graph: create_predict_wstack_graph
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    
    def selfcal_single(vis, model_vis, **kwargs):
        gtsol = solve_gaintable(vis, model_vis, **kwargs)
        vis = apply_gaintable(vis, gtsol, inverse=True, **kwargs)
        return vis
    
    model_vis_graph_list = create_zero_vis_graph_list(vis_graph_list)
    model_vis_graph_list = c_predict_vis_graph(model_vis_graph_list, model_graph, vis_slices=vis_slices, **kwargs)
    
    if global_solution:
        
        global_vis_graph = delayed(visibility_gather_channel, nout=1)(vis_graph_list)
        global_model_vis_graph = delayed(visibility_gather_channel, nout=1)(model_vis_graph_list)
        
        solve_graph = delayed(selfcal_single, pure=True, nout=1)(global_vis_graph, global_model_vis_graph, **kwargs)
        return delayed(visibility_scatter_channel, pure=True, nout=len(vis_graph_list))(solve_graph)
    else:
        return [delayed(selfcal_single, pure=True, nout=1)(vis_graph_list[i], model_vis_graph_list[i], **kwargs)
                for i, _ in enumerate(vis_graph_list)]
