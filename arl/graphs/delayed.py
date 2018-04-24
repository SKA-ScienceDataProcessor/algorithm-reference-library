""" Common functions converted to Dask.execute graphs. `Dask <http://dask.pydata.org/>`_ is a python-based flexible
parallel computing library for analytic computing. Dask.delayed can be used to wrap functions for deferred execution
thus allowing construction of graphs. For example, to build a graph for a major/minor cycle algorithm::

    model_graph = arlexecute.compute(create_image_from_visibility)(vt, npixel=512, cellsize=0.001, npol=1)
    solution_graph = create_solve_image_graph(vt, model_graph=model_graph, psf_graph=psf_graph,
                                            context='timeslice', algorithm='hogbom',
                                            niter=1000, fractional_threshold=0.1,
                                            threshold=1.0, nmajor=3, gain=0.1)
    solution_graph.visualize()

The graph for one vis_graph is executed as follows::

    solution_graph[0].compute()
    
or if a Dask.distributed client is available:

    client.compute(solution_graph)

As well as the specific graphs constructed by functions in this module, there are generic versions in the module
:mod:`arl.pipelines.generic_dask_graphs`.

Construction of the graphs requires that the number of nodes (e.g. w slices or time-slices) be known at construction,
rather than execution. To counteract this, at run time, a given node should be able to act as a no-op. We use None
to denote a null node.

The actual imaging code executed eventually is specified by the context variable (see arl.imaging.imaging)context.
These are the same as executed in the imaging framework.

"""

import collections

import numpy
from dask.distributed import wait

from arl.calibration.calibration_control import calibrate_function
from arl.calibration.operations import apply_gaintable
from arl.data.data_models import Image
from arl.data.parameters import get_parameter
from arl.graphs.execute import arlexecute
from arl.image.deconvolution import deconvolve_cube, restore_cube
from arl.image.gather_scatter import image_scatter_facets, image_gather_facets, image_scatter_channels, \
    image_gather_channels
from arl.image.operations import copy_image, create_empty_image_like
from arl.imaging import normalize_sumwt
from arl.imaging.imaging_context import imaging_context
from arl.imaging.weighting import weight_visibility
from arl.visibility.base import copy_visibility
from arl.visibility.gather_scatter import visibility_gather_channel
from arl.visibility.gather_scatter import visibility_scatter, visibility_gather
from arl.visibility.operations import divide_visibility, integrate_visibility_by_channel


def compute_list(client, graph_list, **kwargs):
    """ Compute all elements in list

    :param client: Client from dask.distributed
    :param graph_list:
    :return: list
    """
    nworkers_initial = len(client.scheduler_info()['workers'])
    futures = client.compute(graph_list, **kwargs)
    wait(futures)
    nworkers_final = len(client.scheduler_info()['workers'])
    # Check that the number of workers has not decreased. On the first call, it seems that
    # Dask can report fewer workers than requested. This is transitory so we only
    # check for decreases.
    # assert nworkers_final >= nworkers_initial, "Lost workers: started with %d, now have %d" % \
    #                                            (nworkers_initial, nworkers_final)
    if nworkers_final < nworkers_initial:
        print("Lost workers: started with %d, now have %d" % (nworkers_initial, nworkers_final))
    return [f.result() for f in futures]


def sum_invert_results(image_list):
    """ Sum a set of invert results with appropriate weighting

    :param image_list: List of [image, sum weights] pairs
    :return: image, sum of weights
    """
    if len(image_list) == 1:
        return image_list[0]
    
    first = True
    sumwt = 0.0
    im = None
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


def remove_sumwt(results):
    """ Remove sumwt term in list of tuples (image, sumwt)
    
    :param results:
    :return: A list of just the dirty images
    """
    return [d[0] for d in results]


def sum_predict_results(results):
    """ Sum a set of predict results of the same shape

    :param results: List of visibilities to be summed
    :return: summed visibility
    """
    sum_results = None
    for result in results:
        if result is not None:
            if sum_results is None:
                sum_results = copy_visibility(result)
            else:
                assert sum_results.data['vis'].shape == result.data['vis'].shape
                sum_results.data['vis'] += result.data['vis']
    
    return sum_results


def create_zero_vis_graph_list(vis_graph_list):
    """ Initialise vis to zero: creates new data holders

    :param vis_graph_list:
    :return: List of vis_graphs
   """
    
    def zero(vis):
        if vis is not None:
            zerovis = copy_visibility(vis)
            zerovis.data['vis'][...] = 0.0
            return zerovis
        else:
            return None
    
    return [arlexecute.execute(zero, pure=True, nout=1)(v) for v in vis_graph_list]


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
    
    return [arlexecute.execute(subtract_vis, pure=True, nout=1)(vis=vis_graph_list[i],
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
    
    def weight_vis(vis, model):
        if vis is not None:
            if model is not None:
                vis, _, _ = weight_visibility(vis, model, weighting=weighting, **kwargs)
                return vis
            else:
                return None
        else:
            return None
    
    return [arlexecute.execute(weight_vis, pure=True, nout=1)(vis_graph_list[i], model_graph[i])
            for i in range(len(vis_graph_list))]


def create_invert_graph(vis_graph_list, template_model_graph, dopsf=False, normalize=True,
                        facets=1, vis_slices=1, context='2d', **kwargs):
    """ Sum results from invert, iterating over the scattered image and vis_graph_list

    :param vis_graph_list:
    :param template_model_graph: Model used to determine image parameters
    :param dopsf: Make the PSF instead of the dirty image
    :param facets: Number of facets
    :param normalize: Normalize by sumwt
    :param vis_slices: Number of slices
    :param context: Imaging context
    :param kwargs: Parameters for functions in graphs
    :return for invert
   """
    
    if not isinstance(template_model_graph, collections.Iterable):
        template_model_graph = [template_model_graph]
    
    c = imaging_context(context)
    vis_iter = c['vis_iterator']
    invert = c['invert']
    inner = c['inner']
    
    if facets % 2 == 0 or facets == 1:
        actual_number_facets = facets
    else:
        actual_number_facets = max(1, (facets - 1))
    
    def gather_image_iteration_results(results, template_model):
        result = create_empty_image_like(template_model)
        i = 0
        sumwt = numpy.zeros([template_model.nchan, template_model.npol])
        for dpatch in image_scatter_facets(result, facets=facets):
            assert i < len(results), "Too few results in gather_image_iteration_results"
            if results[i] is not None:
                assert len(results[i]) == 2, results[i]
                dpatch.data[...] = results[i][0].data[...]
                sumwt += results[i][1]
                i += 1
        return result, sumwt
    
    def invert_ignore_none(vis, model):
        if vis is not None:
            return invert(vis, model, context=context, dopsf=dopsf, normalize=normalize, facets=facets,
                          vis_slices=vis_slices, **kwargs)
        else:
            return create_empty_image_like(model), 0.0
    
    # Loop over all vis_graphs independently
    results_vis_graph_list = list()
    for freqwin, vis_graph in enumerate(vis_graph_list):
        # Create the graph to divide an image into facets. This is by reference.
        facet_graphs = arlexecute.execute(image_scatter_facets, nout=actual_number_facets ** 2)(template_model_graph[
                                                                                                    freqwin],
                                                                                                facets=facets)
        # Create the graph to divide the visibility into slices. This is by copy.
        sub_vis_graphs = arlexecute.execute(visibility_scatter, nout=vis_slices)(vis_graph, vis_iter,
                                                                                 vis_slices=vis_slices)
        
        # Iterate within each vis_graph
        if inner == 'vis':
            vis_results = list()
            for facet_graph in facet_graphs:
                facet_vis_results = list()
                for sub_vis_graph in sub_vis_graphs:
                    facet_vis_results.append(
                        arlexecute.execute(invert_ignore_none, pure=True)(sub_vis_graph, facet_graph))
                vis_results.append(arlexecute.execute(sum_invert_results)(facet_vis_results))
            
            results_vis_graph_list.append(arlexecute.execute(gather_image_iteration_results,
                                                             nout=1)(vis_results, template_model_graph[freqwin]))
        else:
            vis_results = list()
            for sub_vis_graph in sub_vis_graphs:
                facet_vis_results = list()
                for facet_graph in facet_graphs:
                    facet_vis_results.append(
                        arlexecute.execute(invert_ignore_none, pure=True)(sub_vis_graph, facet_graph))
                vis_results.append(arlexecute.execute(gather_image_iteration_results, nout=1)(facet_vis_results,
                                                                                              template_model_graph[
                                                                                                  freqwin]))
            results_vis_graph_list.append(arlexecute.execute(sum_invert_results)(vis_results))
    
    return results_vis_graph_list


def create_predict_graph(vis_graph_list, model_graph, vis_slices=1, facets=1, context='2d', **kwargs):
    """Predict, iterating over both the scattered vis_graph_list and image
    
    The visibility and image are scattered, the visibility is predicted on each part, and then the
    parts are assembled.

    :param vis_graph_list:
    :param model_graph: Model used to determine image parameters
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param facets: Number of facets (per axis)
    :param context:
    :param kwargs: Parameters for functions in graphs
    :return: List of vis_graphs
   """
    
    assert len(vis_graph_list) == len(model_graph), "Model must be the same length as the vis_graph_list"
    
    c = imaging_context(context)
    vis_iter = c['vis_iterator']
    predict = c['predict']
    inner = c['inner']
    
    if facets % 2 == 0 or facets == 1:
        actual_number_facets = facets
    else:
        actual_number_facets = facets - 1
    
    def predict_ignore_none(vis, model):
        if vis is not None:
            return predict(vis, model, context=context, facets=facets, vis_slices=vis_slices, **kwargs)
        else:
            return None
    
    image_results_graph_list = list()
    # Loop over all frequency windows
    for freqwin, vis_graph in enumerate(vis_graph_list):
        # Create the graph to divide an image into facets. This is by reference.
        facet_graphs = arlexecute.execute(image_scatter_facets, nout=actual_number_facets ** 2)(model_graph[freqwin],
                                                                                                facets=facets)
        # Create the graph to divide the visibility into slices. This is by copy.
        sub_vis_graphs = arlexecute.execute(visibility_scatter, nout=vis_slices)(vis_graph, vis_iter, vis_slices)
        
        if inner == 'vis':
            facet_vis_graphs = list()
            # Loop over facets
            for facet_graph in facet_graphs:
                facet_vis_results = list()
                # Loop over sub visibility
                for sub_vis_graph in sub_vis_graphs:
                    facet_vis_graph = arlexecute.execute(predict_ignore_none, pure=True, nout=1)(sub_vis_graph,
                                                                                                 facet_graph)
                    facet_vis_results.append(facet_vis_graph)
                facet_vis_graphs.append(
                    arlexecute.execute(visibility_gather, nout=1)(facet_vis_results, vis_graph, vis_iter))
            # Sum the current sub-visibility over all facets
            image_results_graph_list.append(arlexecute.execute(sum_predict_results)(facet_vis_graphs))
        else:
            facet_vis_graphs = list()
            # Loop over sub visibility
            for sub_vis_graph in sub_vis_graphs:
                facet_vis_results = list()
                # Loop over facets
                for facet_graph in facet_graphs:
                    # Predict visibility for this subvisibility from this facet
                    facet_vis_graph = arlexecute.execute(predict_ignore_none, pure=True, nout=1)(sub_vis_graph,
                                                                                                 facet_graph)
                    facet_vis_results.append(facet_vis_graph)
                # Sum the current sub-visibility over all facets
                facet_vis_graphs.append(arlexecute.execute(sum_predict_results)(facet_vis_results))
            # Sum all sub-visibilties
            image_results_graph_list.append(
                arlexecute.execute(visibility_gather, nout=1)(facet_vis_graphs, vis_graph, vis_iter))
    
    return image_results_graph_list


def create_residual_graph(vis, model_graph, context='2d', **kwargs):
    """ Create a graph to calculate residual image using w stacking and faceting

    :param context: 
    :param vis:
    :param model_graph: Model used to determine image parameters
    :param vis:
    :param model_graph: Model used to determine image parameters
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    model_vis = create_zero_vis_graph_list(vis)
    model_vis = create_predict_graph(model_vis, model_graph, context=context, **kwargs)
    residual_vis = create_subtract_vis_graph_list(vis, model_vis)
    return create_invert_graph(residual_vis, model_graph, dopsf=False, normalize=True, context=context,
                               **kwargs)


def create_restore_graph(model_graph, psf_graph, residual_graph, **kwargs):
    """ Create a graph to calculate the restored image

    :param model_graph: Model graph
    :param psf_graph: PSF graph
    :param residual_graph: Residual graph
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    return [arlexecute.execute(restore_cube)(model_graph[i], psf_graph[i][0], residual_graph[i][0], **kwargs)
            for i, _ in enumerate(model_graph)]


def create_deconvolve_graph(dirty_graph, psf_graph, model_graph, **kwargs):
    """Create a graph for deconvolution, adding to the model

    :param dirty_graph:
    :param psf_graph:
    :param model_graph:
    :param kwargs: Parameters for functions in graphs
    :return: (graph for the deconvolution, graph for the flat)
    """
    
    nchan = len(dirty_graph)
    
    def deconvolve(dirty, psf, model):
        # Gather the channels into one image
        result, _ = deconvolve_cube(dirty, psf, **kwargs)
        if result.data.shape[0] == model.data.shape[0]:
            result.data += model.data
        # Return the cube
        return result
    
    deconvolve_facets = get_parameter(kwargs, 'deconvolve_facets', 1)
    deconvolve_overlap = get_parameter(kwargs, 'deconvolve_overlap', 0)
    deconvolve_taper = get_parameter(kwargs, 'deconvolve_taper', None)
    if deconvolve_overlap > 0:
        deconvolve_number_facets = (deconvolve_facets - 2) ** 2
    else:
        deconvolve_number_facets = deconvolve_facets ** 2
    
    model_graph = arlexecute.execute(image_gather_channels, nout=1)(model_graph)
    
    # Scatter the separate channel images into deconvolve facets and then gather channels for each facet.
    # This avoids constructing the entire spectral cube.
    #    dirty_graph = arlexecute.execute(remove_sumwt, nout=nchan)(dirty_graph)
    scattered_channels_facets_dirty_graph = \
        [arlexecute.execute(image_scatter_facets, nout=deconvolve_number_facets)(d[0], facets=deconvolve_facets,
                                                                                 overlap=deconvolve_overlap,
                                                                                 taper=deconvolve_taper)
         for d in dirty_graph]
    
    # Now we do a transpose and gather
    scattered_facets_graph = [
        arlexecute.execute(image_gather_channels, nout=1)([scattered_channels_facets_dirty_graph[chan][facet]
                                                           for chan in range(nchan)])
        for facet in range(deconvolve_number_facets)]
    
    psf_graph = arlexecute.execute(remove_sumwt, nout=nchan)(psf_graph)
    psf_graph = arlexecute.execute(image_gather_channels, nout=1)(psf_graph)
    
    scattered_model_graph = arlexecute.execute(image_scatter_facets, nout=deconvolve_number_facets)(model_graph,
                                                                                                    facets=deconvolve_facets,
                                                                                                    overlap=deconvolve_overlap)
    
    # Now do the deconvolution for each facet
    scattered_results_graph = [arlexecute.execute(deconvolve, nout=1)(d, psf_graph, m)
                               for d, m in zip(scattered_facets_graph, scattered_model_graph)]
    
    # Gather the results back into one image, correcting for overlaps as necessary. The taper function is is used to
    # feather the facets together
    gathered_results_graph = arlexecute.execute(image_gather_facets, nout=1)(scattered_results_graph, model_graph,
                                                                             facets=deconvolve_facets,
                                                                             overlap=deconvolve_overlap,
                                                                             taper=deconvolve_taper)
    flat_graph = arlexecute.execute(image_gather_facets, nout=1)(scattered_results_graph, model_graph,
                                                                 facets=deconvolve_facets, overlap=deconvolve_overlap,
                                                                 taper=deconvolve_taper, return_flat=True)
    
    return arlexecute.execute(image_scatter_channels, nout=nchan)(gathered_results_graph, subimages=nchan), flat_graph


def create_deconvolve_channel_graph(dirty_graph, psf_graph, model_graph, subimages,
                                    **kwargs):
    """Create a graph for deconvolution by channels, adding to the model

    Does deconvolution channel by channel.
    :param subimages: 
    :param dirty_graph:
    :param psf_graph: Must be the size of a facet
    :param model_graph: Current model
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    
    def deconvolve_subimage(dirty, psf):
        assert isinstance(dirty, Image)
        assert isinstance(psf, Image)
        comp = deconvolve_cube(dirty, psf, **kwargs)
        return comp[0]
    
    def add_model(sum_model, model):
        assert isinstance(output, Image)
        assert isinstance(model, Image)
        sum_model.data += model.data
        return sum_model
    
    output = arlexecute.execute(create_empty_image_like, nout=1, pure=True)(model_graph)
    dirty_graphs = arlexecute.execute(image_scatter_channels, nout=subimages, pure=True)(dirty_graph[0],
                                                                                         subimages=subimages)
    results = [arlexecute.execute(deconvolve_subimage)(dirty_graph, psf_graph[0])
               for dirty_graph in dirty_graphs]
    result = arlexecute.execute(image_gather_channels, nout=1, pure=True)(results, output, subimages=subimages)
    return arlexecute.execute(add_model, nout=1, pure=True)(result, model_graph)


def create_calibrate_graph_list(vis_graph_list, model_vis_graph_list, calibration_context='TG', global_solution=True,
                                **kwargs):
    """ Create a set of graphs for (optionally global) calibration of a list of visibilities

    If global solution is true then visibilities are gathered to a single visibility data set which is then
    self-calibrated. The resulting gaintable is then effectively scattered out for application to each visibility
    set. If global solution is false then the solutions are performed locally.

    :param vis_graph_list:
    :param model_vis_graph_list:
    :param calibration_context: String giving terms to be calibrated e.g. 'TGB'
    :param global_solution: Solve for global gains
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    
    def solve_and_apply(vis, modelvis=None):
        return calibrate_function(vis, modelvis, calibration_context=calibration_context, **kwargs)[0]
    
    if global_solution:
        point_vis_graph_list = [arlexecute.execute(divide_visibility, nout=len(vis_graph_list))(vis_graph_list[i],
                                                                                                model_vis_graph_list[i])
                                for i, _ in enumerate(vis_graph_list)]
        global_point_vis_graph = arlexecute.execute(visibility_gather_channel, nout=1)(point_vis_graph_list)
        global_point_vis_graph = arlexecute.execute(integrate_visibility_by_channel, nout=1)(global_point_vis_graph)
        # This is a global solution so we only compute one gain table
        _, gt_graph = arlexecute.execute(solve_and_apply, pure=True, nout=2)(global_point_vis_graph, **kwargs)
        return [arlexecute.execute(apply_gaintable, nout=len(vis_graph_list))(v, gt_graph, inverse=True)
                for v in vis_graph_list]
    else:
        
        return [
            arlexecute.execute(solve_and_apply, nout=len(vis_graph_list))(vis_graph_list[i], model_vis_graph_list[i])
            for i, v in enumerate(vis_graph_list)]
