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

Construction of the graphs requires that the number of nodes (e.g. w slices or time-slices) be known at construction,
rather than execution. To counteract this, at run time, a given node should be able to act as a no-op. This is a
workaround only.

"""

import numpy
from dask import delayed
from dask.distributed import wait

from arl.calibration.operations import apply_gaintable
from arl.calibration.solvers import solve_gaintable
from arl.data.data_models import Image, BlockVisibility
from arl.image.deconvolution import deconvolve_cube
from arl.image.gather_scatter import image_scatter_facets, image_gather_facets, image_scatter_channels, \
    image_gather_channels
from arl.image.operations import copy_image, create_empty_image_like
from arl.imaging import normalize_sumwt
from arl.imaging.imaging_context import imaging_context
from arl.imaging.weighting import weight_visibility
from arl.visibility.base import copy_visibility, create_visibility_from_rows
from arl.visibility.coalesce import coalesce_visibility
from arl.visibility.gather_scatter import visibility_scatter
from arl.visibility.operations import divide_visibility, integrate_visibility_by_channel


def compute_list(client, graph_list, nodes=None, **kwargs):
    """ Compute all elements in list

    :param graph_list:
    :param nodes: List of nodes.
    :return: list
    """
    nworkers_initial = len(client.scheduler_info()['workers'])
    futures = client.compute(graph_list, **kwargs)
    wait(futures)
    nworkers_final = len(client.scheduler_info()['workers'])
    # Check that the number of workers has not decreased. On the first call, it seems that
    # Dask can report fewer workers than requested. This is transitory so we only
    # check for decreases.
    assert nworkers_final >= nworkers_initial, "Lost workers: started with %d, now have %d" % \
                                               (nworkers_initial, nworkers_final)
    return [f.result() for f in futures]


def sum_invert_results(image_list):
    """ Sum a set of invert results with appropriate weighting

    :param image_list: List of [image, sum weights] pairs
    :return: image, sum of weights
    """
    first = True
    for i, arg in enumerate(image_list):
        if arg is not None:
            if isinstance(arg[1], numpy.ndarray):
                scale = arg[1][..., numpy.newaxis, numpy.newaxis]
            else:
                scale = arg[1]
            if first:
                im = copy_image(arg[0])
                im.data *= scale
                sumwt = arg[1]
                first = False
            else:
                im.data += scale * arg[0].data
                sumwt += arg[1]
    
    assert not first, "No invert results"
    
    im = normalize_sumwt(im, sumwt)
    return im, sumwt


def sum_predict_results(results):
    """ Sum a set of predict results

    :param vis_list: List of visibilities
    :return: summed visibility
    """
    sum_results = None
    print("In to sum_predict_results: ", results)
    for result in results:
        if result is not None:
            if sum_results is None:
                sum_results = copy_visibility(result)
            else:
                sum_results.data['vis'] += result.data['vis']
    print("Out from sum_predict_results: ", sum_results)

    return sum_results

def create_zero_vis_graph_list(vis_graph_list):
    """ Initialise vis to zero: creates new data holders

    :param vis_graph_list:
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


def create_subtract_vis_graph_list(vis_graph_list, model_vis_graph_list):
    """ Initialise vis to zero

    :param vis_graph_list:
    :param model_vis_graph_list: Model to be subtracted
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


def create_weight_vis_graph_list(vis_graph_list, model_graph, weighting='uniform', **kwargs):
    """ Weight the visibility data

    :param vis_graph_list:
    :param model_graph: Model required to determine weighting parameters
    :param weighting: Type of weighting
    :param kwargs: Parameters for functions in graphs
    :return: List of vis_graphs
   """
    
    def weight_vis(vis, model, weighting):
        if vis is not None:
            if model is not None:
                vis, _, _ = weight_visibility(vis, model, weighting=weighting, **kwargs)
                return vis
            else:
                return None
        else:
            return None
    
    return [delayed(weight_vis, pure=True, nout=1)(vis_graph_list[i], model_graph, weighting)
            for i in range(len(vis_graph_list))]


def create_invert_graph(vis_graph_list, template_model_graph: delayed, dopsf=False, normalize=True,
                        facets=1, vis_slices=1, context='2d', **kwargs) -> delayed:
    """ Sum results from invert, iterating over the scattered image and vis_graph_list

    :param vis_graph_list:
    :param template_model_graph: Model used to determine image parameters
    :param c_invert_vis_scatter_graph: Function to create invert graphs
    :param dopsf: Make the PSF instead of the dirty image
    :param facets: Number of facets
    :param kwargs: Parameters for functions in graphs
    :return: delayed for invert
   """
    c = imaging_context(context)
    invert = c['invert']
    image_iter = c['image_iterator']
    vis_iter = c['vis_iterator']
    
    def scatter_vis(vis, vis_slices, vis_iter, **kwargs):
        if isinstance(vis, BlockVisibility):
            avis = coalesce_visibility(vis, **(kwargs))
        else:
            avis = vis
        return visibility_scatter(avis, vis_iter=vis_iter, vis_slices=vis_slices, **kwargs)
    
    def scatter_image(im, facets, image_iter, **kwargs):
        return [subim for subim in image_iter(im, facets=facets, **kwargs)]

    def gather_invert_results(results, template_model, facets, image_iter, **kwargs):
        result = create_empty_image_like(template_model)
        i = 0
        for dpatch in image_iter(result, facets=facets, **kwargs):
            if results[i] is not None:
                dpatch.data[...] = results[i][0].data[...]
                i+=1
        return result, results[0][1]

    def invert_ignore_None(vis, model, *args, **kwargs):
        if vis is not None:
            return invert(vis, model, *args, context=context, dopsf=dopsf, normalize=normalize, **kwargs)
        else:
            return create_empty_image_like(model), 0.0
    
    # Scatter the model in e.g. facets
    model_graphs = delayed(scatter_image, nout=facets ** 2)(template_model_graph, facets=facets, image_iter=image_iter)
    # Loop over all vis_graphs independently
    results_vis_graph_list = list()
    for vis_graph in vis_graph_list:
        sub_vis_graphs = delayed(scatter_vis, nout=vis_slices)(vis_graph, vis_slices=vis_slices,
                                                               vis_iter=vis_iter, **kwargs)
        # Iterate within each vis_graph
        image_graphs = list()
        # Iterate within each model_graph
        for sub_model_graph in model_graphs:
            sub_model_results = list()
            for sub_vis_graph in sub_vis_graphs:
                sub_model_results.append(delayed(invert_ignore_None, pure=True, nout=1) \
                                        (sub_vis_graph, sub_model_graph, **kwargs))
            image_graphs.append(delayed(sum_invert_results, nout=1)(sub_model_results))
            
        results_vis_graph_list.append(delayed(gather_invert_results)(image_graphs, template_model_graph, facets,
                                                                     image_iter))
    
    return results_vis_graph_list


def create_predict_graph(vis_graph_list, model_graph: delayed, vis_slices=1, facets=1, context='2d', **kwargs):
    """Predict, iterating over both the scattered vis_graph_list and image

    :param vis_graph_list:
    :param model_graph: Model used to determine image parameters
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param predict: Predict function
    :param vis_scatter: Scatter function e.g. visibility_scatter_w
    :param vis_gather: Gatherer function e.g. visibility_gather_w
    :param kwargs: Parameters for functions in graphs
    :return: List of vis_graphs
   """
    c = imaging_context(context)
    predict = c['predict']
    image_iter = c['image_iterator']
    vis_iter = c['vis_iterator']
    
    def predict_ignore_None(vis, model, **kwargs):
        if vis is not None:
            predicted = copy_visibility(vis)
            predicted = predict(predicted, model, **kwargs)
            return predicted
        else:
            return None
    
    def gather_vis(results, vis, vis_slices, **kwargs):
        i = 0
        for rows in vis_iter(vis, vis_slices=vis_slices, **kwargs):
            if rows is not None:
                print(i, numpy.sum(rows))
                vis.data['vis'][rows][...] = results[i].data['vis'][...]
                i+=1
        return vis

    def scatter_vis(vis, vis_slices, vis_iter, **kwargs):
        if isinstance(vis, BlockVisibility):
            avis = coalesce_visibility(vis, **(kwargs))
        else:
            avis = vis
        results = visibility_scatter(avis, vis_iter=vis_iter, vis_slices=vis_slices, **kwargs)
        return results

    def scatter_image(im, facets, **kwargs):
        return [subim for subim in image_iter(im, facets=facets, **kwargs)]
    
    model_graphs = delayed(scatter_image, nout=facets ** 2)(model_graph, facets=facets)
    
    results_vis_graph_list = list()
    for vis_graph in vis_graph_list:
        sub_vis_graphs = delayed(scatter_vis, nout=vis_slices)(vis_graph, vis_slices=vis_slices,
                                                               vis_iter=vis_iter, **kwargs)
        vis_graphs = list()
        for sub_model_graph in model_graphs:
            sub_model_results = list()
            for sub_vis_graph in sub_vis_graphs:
                sub_model_results.append(delayed(predict_ignore_None, pure=True, nout=1) \
                    (sub_vis_graph, sub_model_graph, **kwargs))
            vis_graphs.append(delayed(sum_predict_results)(sub_model_results))

        results_vis_graph_list.append(delayed(gather_vis, nout=1)(vis_graphs, vis_graph,
                                                                  vis_slices=vis_slices, **kwargs))
    return results_vis_graph_list


def create_residual_graph(vis, model_graph: delayed, context='2d', **kwargs) -> delayed:
    """ Create a graph to calculate residual image using w stacking and faceting

    :param vis:
    :param model_graph: Model used to determine image parameters
    :param vis:
    :param model_graph: Model used to determine image parameters
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param facets: Number of facets (in both x and y axes)
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    model_vis = create_zero_vis_graph_list(vis)
    model_vis = create_predict_graph(model_vis, model_graph, context=context, **kwargs)
    residual_vis = create_subtract_vis_graph_list(vis, model_vis)
    return create_invert_graph(residual_vis, model_graph, dopsf=False, normalize=True, context=context,
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


def create_deconvolve_facet_graph(dirty_graph: delayed, psf_graph: delayed, model_graph: delayed,
                                  facets=1, **kwargs) -> delayed:
    """Create a graph for deconvolution by subimages, adding to the model
    
    Does deconvolution subimage by subimage. Currently does nothing very sensible about the
    edges.

    :param dirty_graph:
    :param psf_graph:
    :param model_graph: Current model
    :param subimages: Number of subimages
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    
    def deconvolve_subimage(dirty, psf, **kwargs):
        assert isinstance(dirty, Image)
        assert isinstance(psf, Image)
        result = deconvolve_cube(dirty, psf, **kwargs)
        return result[0]
    
    def add_model(output, model):
        assert isinstance(output, Image)
        assert isinstance(model, Image)
        output.data += model.data
        return output
    
    output = delayed(create_empty_image_like, nout=1, pure=True)(model_graph)
    dirty_graphs = delayed(image_scatter_facets, nout=facets * facets, pure=True)(dirty_graph[0], facets=facets)
    results = [delayed(deconvolve_subimage)(dirty_graph, psf_graph[0], **kwargs)
               for dirty_graph in dirty_graphs]
    result = delayed(image_gather_facets, nout=1, pure=True)(results, output, facets=facets)
    return delayed(add_model, nout=1, pure=True)(result, model_graph)


def create_deconvolve_channel_graph(dirty_graph: delayed, psf_graph: delayed, model_graph: delayed, subimages,
                                    **kwargs) -> delayed:
    """Create a graph for deconvolution by channels, adding to the model

    Does deconvolution channel by channel.
    :param dirty_graph:
    :param psf_graph: Must be the size of a facet
    :param model_graph: Current model
    :param facets: Number of facets on each axis
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    
    def deconvolve_subimage(dirty, psf, **kwargs):
        assert isinstance(dirty, Image)
        assert isinstance(psf, Image)
        result = deconvolve_cube(dirty, psf, **kwargs)
        return result[0]
    
    def add_model(output, model):
        assert isinstance(output, Image)
        assert isinstance(model, Image)
        output.data += model.data
        return output
    
    output = delayed(create_empty_image_like, nout=1, pure=True)(model_graph)
    dirty_graphs = delayed(image_scatter_channels, nout=subimages, pure=True)(dirty_graph[0], subimages=subimages)
    results = [delayed(deconvolve_subimage)(dirty_graph, psf_graph[0], **kwargs)
               for dirty_graph in dirty_graphs]
    result = delayed(image_gather_channels, nout=1, pure=True)(results, output, subimages=subimages)
    return delayed(add_model, nout=1, pure=True)(result, model_graph)


def create_selfcal_graph_list(vis_graph_list, model_graph: delayed, c_predict_graph,
                              vis_slices, global_solution=True, **kwargs):
    """ Create a set of graphs for (optionally global) selfcalibration of a list of visibilities

    If global solution is true then visibilities are gathered to a single visibility data set which is then
    self-calibrated. The resulting gaintable is then effectively scattered out for application to each visibility
    set. If global solution is false then the solutions are performed locally.

    :param vis_graph_list:
    :param model_graph:
    :param c_predict_graph: Function to create prediction graphs
    :param vis_slices:
    :param global_solution: Solve for global gains
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    
    model_vis_graph_list = create_zero_vis_graph_list(vis_graph_list)
    model_vis_graph_list = c_predict_graph(model_vis_graph_list, model_graph, vis_slices=vis_slices, **kwargs)
    return create_calibrate_graph_list(vis_graph_list, model_vis_graph_list, **kwargs)


def create_calibrate_graph_list(vis_graph_list, model_vis_graph_list, global_solution=True, **kwargs):
    """ Create a set of graphs for (optionally global) calibration of a list of visibilities

    If global solution is true then visibilities are gathered to a single visibility data set which is then
    self-calibrated. The resulting gaintable is then effectively scattered out for application to each visibility
    set. If global solution is false then the solutions are performed locally.

    :param vis_graph_list:
    :param model_vis_graph_list:
    :param vis_slices:
    :param global_solution: Solve for global gains
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    
    if global_solution:
        point_vis_graph_list = [delayed(divide_visibility, nout=len(vis_graph_list))(vis_graph_list[i],
                                                                                     model_vis_graph_list[i])
                                for i, _ in enumerate(vis_graph_list)]
        global_point_vis_graph = delayed(visibility_gather_channel, nout=1)(point_vis_graph_list)
        global_point_vis_graph = delayed(integrate_visibility_by_channel, nout=1)(global_point_vis_graph)
        # This is a global solution so we only get one gain table
        gt_graph = delayed(solve_gaintable, pure=True, nout=1)(global_point_vis_graph, **kwargs)
        return [delayed(apply_gaintable, nout=len(vis_graph_list))(v, gt_graph, inverse=True)
                for v in vis_graph_list]
    else:
        def solve_and_apply(vis, modelvis, **kwargs):
            gt = solve_gaintable(vis, modelvis, **kwargs)
            return apply_gaintable(vis, gt, **kwargs)
        
        return [delayed(solve_and_apply, nout=len(vis_graph_list))(vis_graph_list[i], model_vis_graph_list[i],
                                                                   inverse=True)
                for i, v in enumerate(vis_graph_list)]
