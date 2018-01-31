""" Common functions converted to Dask.delayed graphs. `Dask <http://dask.pydata.org/>`_ is a python-based flexible
parallel computing library for analytic computing. Dask.delayed can be used to wrap functions for deferred execution
thus allowing construction of graphs. For example, to build a graph for a major/minor cycle algorithm::

    model_graph = delayed(create_image_from_visibility)(vt, npixel=512, cellsize=0.001, npol=1)
    solution_graph = create_solve_image_graph(vt, model_graph=model_graph, psf_graph=psf_graph,
                                            context='timslice', algorithm='hogbom',
                                            niter=1000, fractional_threshold=0.1,
                                            threshold=1.0, nmajor=3, gain=0.1)
    solution_graph.visualize()

The graph for one vis_graph is executed as follows::

    solution_graph[0].compute()
    
or if a Dask.dsitributed client is available:

    client.compute(solution_graph)

As well as the specific graphs constructed by functions in this module, there are generic versions in the module
:mod:`arl.pipelines.generic_dask_graphs`.

Note that all parameters here should be passed using the kwargs mechanism. The exceptions
are those needed to define the size of a graph. Since delayed graphs are not Iterable
by default, it is necessary to use the nout= parameter to delayed to specify the
graph size.

Construction of the graphs requires that the number of nodes (e.g. w slices or time-slices) be known at construction,
rather than execution. To counteract this, at run time, a given node should be able to act as a no-op. We use None
to denote a null node.

The actual imaging code executed eventually is specified by the context variable (see arl.imaging.imaging)context.
These are the same as executed in the imaging framework.

"""

import numpy
from dask import delayed
from dask.distributed import wait

from arl.calibration.operations import apply_gaintable
from arl.calibration.solvers import solve_gaintable
from arl.calibration.calibration_control import calibrate_function
from arl.data.data_models import Image, BlockVisibility
from arl.data.parameters import get_parameter
from arl.image.deconvolution import deconvolve_cube, restore_cube
from arl.image.gather_scatter import image_scatter_facets, image_gather_facets, image_scatter_channels, \
    image_gather_channels
from arl.image.operations import copy_image, create_empty_image_like
from arl.imaging import normalize_sumwt
from arl.imaging.imaging_context import imaging_context
from arl.imaging.weighting import weight_visibility
from arl.visibility.base import copy_visibility, create_visibility_from_rows
from arl.visibility.coalesce import coalesce_visibility, decoalesce_visibility
from arl.visibility.gather_scatter import visibility_gather_channel
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
    assert nworkers_final >= nworkers_initial, "Lost workers: started with %d, now have %d" % \
                                               (nworkers_initial, nworkers_final)
    return [f.result() for f in futures]


def sum_invert_results(image_list):
    """ Sum a set of invert results with appropriate weighting

    :param image_list: List of [image, sum weights] pairs
    :return: image, sum of weights
    """
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


def sum_predict_results(results):
    """ Sum a set of predict results

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
    
    return [delayed(zero, pure=True, nout=1)(v) for v in vis_graph_list]


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
    
    def weight_vis(vis, model):
        if vis is not None:
            if model is not None:
                vis, _, _ = weight_visibility(vis, model, weighting=weighting, **kwargs)
                return vis
            else:
                return None
        else:
            return None
    
    return [delayed(weight_vis, pure=True, nout=1)(vis_graph_list[i], model_graph)
            for i in range(len(vis_graph_list))]


def create_invert_graph(vis_graph_list, template_model_graph: delayed, dopsf=False, normalize=True,
                        facets=1, vis_slices=None, context='2d', **kwargs) -> delayed:
    """ Sum results from invert, iterating over the scattered image and vis_graph_list

    :param vis_graph_list:
    :param template_model_graph: Model used to determine image parameters
    :param dopsf: Make the PSF instead of the dirty image
    :param facets: Number of facets
    :param normalize: Normalize by sumwt
    :param vis_slices: Number of slices
    :param context: Imaging context
    :param kwargs: Parameters for functions in graphs
    :return: delayed for invert
   """
    c = imaging_context(context)
    image_iter = c['image_iterator']
    vis_iter = c['vis_iterator']
    invert = c['invert']
    inner = c['inner']
    
    def scatter_vis(vis):
        if isinstance(vis, BlockVisibility):
            avis = coalesce_visibility(vis, **kwargs)
        else:
            avis = vis
        result = [create_visibility_from_rows(avis, rows) for rows in vis_iter(avis, vis_slices=vis_slices, **kwargs)]
        assert len(result) == vis_slices, "result %s, vis_slices %d" % (str(result), vis_slices)
        return result
    
    def scatter_image_iteration(im):
        return [subim for subim in image_iter(im, facets=facets, **kwargs)]
    
    def gather_image_iteration_results(results, template_model):
        result = create_empty_image_like(template_model)
        i = 0
        sumwt = numpy.zeros([template_model.nchan, template_model.npol])
        for dpatch in image_iter(result, facets=facets, **kwargs):
            assert i < len(results), "Too few results in gather_image_iteration_results"
            if results[i] is not None:
                assert len(results[i]) == 2, results[i]
                dpatch.data[...] = results[i][0].data[...]
                sumwt += results[i][1]
                i += 1
        return result, sumwt
    
    def invert_ignore_none(vis, model):
        if vis is not None:
            return invert(vis, model, context=context, dopsf=dopsf, normalize=normalize, **kwargs)
        else:
            return create_empty_image_like(model), 0.0
    
    # Loop over all vis_graphs independently
    
    results_vis_graph_list = list()
    for freqwin, vis_graph in enumerate(vis_graph_list):
        sub_vis_graphs = delayed(scatter_vis, nout=vis_slices)(vis_graph)
        model_graphs = delayed(scatter_image_iteration, nout=facets ** 2)(template_model_graph[freqwin])
        
        # Iterate within each vis_graph
        if inner == 'vis':
            model_results = list()
            # Scatter the model in e.g. facets
            for model_graph in model_graphs:
                model_vis_results = list()
                for sub_vis_graph in sub_vis_graphs:
                    model_vis_results.append(delayed(invert_ignore_none, pure=True)(sub_vis_graph, model_graph))
                model_results.append(delayed(sum_invert_results)(model_vis_results))
            results_vis_graph_list.append(delayed(gather_image_iteration_results)(model_results,
                                                                                  template_model_graph[freqwin]))
        else:
            vis_results = list()
            for sub_vis_graph in sub_vis_graphs:
                model_vis_results = list()
                for model_graph in model_graphs:
                    model_vis_results.append(delayed(invert_ignore_none, pure=True)(sub_vis_graph, model_graph))
                vis_results.append(delayed(gather_image_iteration_results)(model_vis_results,
                                                                           template_model_graph[freqwin]))
            results_vis_graph_list.append(delayed(sum_invert_results)(vis_results))
    
    return results_vis_graph_list


def create_predict_graph(vis_graph_list, model_graph: delayed, vis_slices=1, facets=1, context='2d', **kwargs):
    """Predict, iterating over both the scattered vis_graph_list and image

    :param facets: 
    :param context: 
    :param vis_graph_list:
    :param model_graph: Model used to determine image parameters
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param kwargs: Parameters for functions in graphs
    :return: List of vis_graphs
   """
    c = imaging_context(context)
    image_iter = c['image_iterator']
    vis_iter = c['vis_iterator']
    predict = c['predict']
    inner = c['inner']
    
    def predict_ignore_none(vis, model):
        if vis is not None:
            predicted = copy_visibility(vis)
            predicted = predict(predicted, model, context=context, **kwargs)
            return predicted
        else:
            return None
    
    def gather_vis(results, vis):
        # Gather across the visibility iteration axis
        assert vis is not None
        if isinstance(vis, BlockVisibility):
            avis = coalesce_visibility(vis, **kwargs)
        else:
            avis = vis
        for i, rows in enumerate(vis_iter(avis, vis_slices=vis_slices, **kwargs)):
            assert i < len(results), "Insufficient results for the gather"
            if rows is not None and results[i] is not None:
                avis.data['vis'][rows] = results[i].data['vis']
        
        if isinstance(vis, BlockVisibility):
            return decoalesce_visibility(avis, **kwargs)
        else:
            return avis
    
    def scatter_vis(vis):
        # Scatter along the visibility iteration axis
        if isinstance(vis, BlockVisibility):
            avis = coalesce_visibility(vis, **kwargs)
        else:
            avis = vis
        result = [create_visibility_from_rows(avis, rows) for rows in vis_iter(avis, vis_slices=vis_slices, **kwargs)]
        return result
    
    def scatter_image(im):
        # Scatter across image iteration
        return [subim for subim in image_iter(im, facets=facets, **kwargs)]
    
    results_vis_graph_list = list()
    for freqwin, vis_graph in enumerate(vis_graph_list):
        sub_model_graphs = delayed(scatter_image, nout=facets ** 2)(model_graph[freqwin])
        sub_vis_graphs = delayed(scatter_vis, nout=vis_slices)(vis_graph)
        vis_graphs = list()
        for sub_model_graph in sub_model_graphs:
            sub_model_results = list()
            for sub_vis_graph in sub_vis_graphs:
                model_vis_graph = delayed(predict_ignore_none, pure=True, nout=1)(sub_vis_graph,
                                                                                  sub_model_graph)
                sub_model_results.append(model_vis_graph)
            vis_graphs.append(delayed(gather_vis, nout=1)(sub_model_results, vis_graph))
        results_vis_graph_list.append(delayed(sum_predict_results)(vis_graphs))
    
    return results_vis_graph_list


def create_residual_graph(vis, model_graph: delayed, context='2d', **kwargs) -> delayed:
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


def create_restore_graph(model_graph: delayed, psf_graph, residual_graph, **kwargs) -> delayed:
    """ Create a graph to calculate the restored image

    :param model_graph: Model graph
    :param psf_graph: PSF graph
    :param residual_graph: Residual graph
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    return [delayed(restore_cube)(model_graph[i], psf_graph[i][0], residual_graph[i][0], **kwargs)
            for i, _ in enumerate(model_graph)]


def create_deconvolve_graph(dirty_graph: delayed, psf_graph: delayed, model_graph: delayed,
                            **kwargs) -> delayed:
    """Create a graph for deconvolution, adding to the model

    :param dirty_graph:
    :param psf_graph:
    :param model_graph:
    :param nchan: Number of channels
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    nchan = get_parameter(kwargs, "nchan", 1)

    def remove_sumwt(dirty_list):
        return [d[0] for d in dirty_list]
    
    def make_cube_and_deconvolve(dirty, psf, model):
        dirty_cube = image_gather_channels(remove_sumwt(dirty), subimages=nchan)
        psf_cube = image_gather_channels(remove_sumwt(psf), subimages=nchan)
        model_cube = image_gather_channels(model, subimages=nchan)
        result = deconvolve_cube(dirty_cube, psf_cube, **kwargs)
        result[0].data += model_cube.data
        return image_scatter_channels(result[0], nchan)
    

    def deconvolve(dirty, psf, model):
        result = deconvolve_cube(dirty[0], psf[0], **kwargs)
        result[0].data += model.data
        return result[0]

    algorithm = get_parameter(kwargs, "algorithm", 'mmclean')
    if algorithm == "mmclean" and nchan > 1:
        return delayed(make_cube_and_deconvolve, nout=nchan)(dirty_graph, psf_graph, model_graph)
    else:
        return [delayed(deconvolve, pure=True, nout=nchan)(dirty_graph[i], psf_graph[i], model_graph[i])
                for i, _ in enumerate(dirty_graph)]


def create_deconvolve_facet_graph(dirty_graph: delayed, psf_graph: delayed, model_graph: delayed,
                                  facets=1, **kwargs) -> delayed:
    """Create a graph for deconvolution by subimages, adding to the model
    
    Does deconvolution subimage by subimage. Currently does nothing very sensible about the
    edges.

    :param facets: 
    :param dirty_graph:
    :param psf_graph:
    :param model_graph: Current model
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    
    def deconvolve_subimage(dirty, psf, model):
        assert isinstance(dirty, Image)
        assert isinstance(psf, Image)
        assert isinstance(model, Image)
        comp = deconvolve_cube(dirty, psf, **kwargs)
        comp[0].data += model.data
        return comp[0]
    
    output = delayed(create_empty_image_like, nout=1, pure=True)(model_graph)
    dirty_graphs = delayed(image_scatter_facets, nout=facets * facets, pure=True)(dirty_graph[0], facets=facets)
    results = [delayed(deconvolve_subimage)(dirty_graph[i], psf_graph[i], model_graph[i])
               for i, _ in enumerate(dirty_graphs)]
    return delayed(image_gather_facets, nout=1, pure=True)(results, output, facets=facets)


def create_deconvolve_channel_graph(dirty_graph: delayed, psf_graph: delayed, model_graph: delayed, subimages,
                                    **kwargs) -> delayed:
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
    
    output = delayed(create_empty_image_like, nout=1, pure=True)(model_graph)
    dirty_graphs = delayed(image_scatter_channels, nout=subimages, pure=True)(dirty_graph[0], subimages=subimages)
    results = [delayed(deconvolve_subimage)(dirty_graph, psf_graph[0])
               for dirty_graph in dirty_graphs]
    result = delayed(image_gather_channels, nout=1, pure=True)(results, output, subimages=subimages)
    return delayed(add_model, nout=1, pure=True)(result, model_graph)


def create_selfcal_graph_list(vis_graph_list, model_graph: delayed, c_predict_graph,
                              vis_slices, **kwargs):
    """ Create a set of graphs for (optionally global) selfcalibration of a list of visibilities

    If global solution is true then visibilities are gathered to a single visibility data set which is then
    self-calibrated. The resulting gaintable is then effectively scattered out for application to each visibility
    set. If global solution is false then the solutions are performed locally.

    :param vis_graph_list:
    :param model_graph:
    :param c_predict_graph: Function to create prediction graphs
    :param vis_slices:
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
    :param global_solution: Solve for global gains
    :param kwargs: Parameters for functions in graphs
    :return:
    """

    def solve_and_apply(vis, modelvis=None):
        return calibrate_function(vis, modelvis, **kwargs)[0]

    if global_solution:
        point_vis_graph_list = [delayed(divide_visibility, nout=len(vis_graph_list))(vis_graph_list[i],
                                                                                     model_vis_graph_list[i])
                                for i, _ in enumerate(vis_graph_list)]
        global_point_vis_graph = delayed(visibility_gather_channel, nout=1)(point_vis_graph_list)
        global_point_vis_graph = delayed(integrate_visibility_by_channel, nout=1)(global_point_vis_graph)
        # This is a global solution so we only get one gain table
        _, gt_graph = delayed(solve_and_apply, pure=True, nout=2)(global_point_vis_graph, **kwargs)
        return [delayed(apply_gaintable, nout=len(vis_graph_list))(v, gt_graph, inverse=True)
                for v in vis_graph_list]
    else:
        
        return [delayed(solve_and_apply, nout=len(vis_graph_list))(vis_graph_list[i], model_vis_graph_list[i])
                for i, v in enumerate(vis_graph_list)]
