import collections

from dask import delayed

from arl.calibration.solvers import solve_gaintable
from arl.data.data_models import BlockVisibility
from arl.data.parameters import get_parameter
from arl.fourier_transforms.ftprocessor import invert_2d, residual_image, \
    predict_skycomponent_blockvisibility, \
    invert_timeslice_single, predict_timeslice_single, normalize_sumwt
from arl.image.deconvolution import deconvolve_cube, restore_cube
from arl.visibility.iterators import vis_timeslice_iter
from arl.visibility.operations import create_visibility_from_rows, copy_visibility


def create_invert_graph(vt, model_graph, dopsf=False, normalize=True, invert_single=invert_2d,
                        iterator=vis_timeslice_iter, **kwargs):
    """ Create a graph to perform an invert
    
    The graph is constructed by iteration across the visibility
    
    :param vt: Visibility
    :param model_graph: Model graph
    :param dopsf: Make a PSF?
    :param normalize: Normalise the inversion?
    :param invert_single: Function to do a single invert
    :param iterator: visibility iterator
    :returns: Graph in dask.delayed format
    """
    
    def accumulate_results(results, normalize=normalize):
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
    
    for rows in iterator(vt, **kwargs):
        v = copy_visibility(create_visibility_from_rows(vt, rows))
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
        result = deconvolve_cube(dirty, psf, **kwargs)[0]
        result.data += model.data
        return result
    
    return delayed(deconvolve_model_only, pure=True)(dirty_graph[0], psf_graph[0], model_graph, **kwargs)


def create_residual_graph(vis, model_graph, iterator=vis_timeslice_iter, **kwargs):
    """ Create a graph to calculate residual visibility

    :param vt: Visibility
    :param model_graph: Model graph
    :param iterator: visibility iterator
    :param kwargs: Other arguments for ftprocessor.residual
    :returns: residual visibility graph
    """
    
    def accumulate_results(results, rowses):
        
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
    :param model_graph: Model graph
    :param residual_graph: Residual graph used for residual visibilities
    :param invert_graph: Invert graph used for PSF
    :param iterator: visibility iterator
    :param kwargs: Other arguments for ftprocessor.residual
    :returns: residual visibility graph
    """
    psf_graph = create_invert_graph(vis, model_graph, dopsf=True, **kwargs)
    
    res_graph_list = list()
    model_graph_list = list()
    
    nmajor = get_parameter(kwargs, "nmajor", 5)
    
    res_graph_list.append(create_residual_graph(vis, model_graph, **kwargs))
    model_graph_list.append(create_deconvolve_graph(res_graph_list[-1], psf_graph, model_graph, **kwargs))
    
    for cycle in range(1, nmajor):
        res_graph_list.append(create_residual_graph(vis, model_graph_list[-1], **kwargs))
        model_graph_list.append(create_deconvolve_graph(res_graph_list[-1], psf_graph,
                                                        model_graph_list[cycle - 1], **kwargs))
    
    return model_graph_list[-1]


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
    
    :param vis: Actual visibility
    :param model_graph: graph for the model
    :param invert_residual: function used in ftprocessor.residual
    :param predict_residual: function used in ftprocessor.residual
    :param kwargs:
    :return:
    """
    psf_graph = create_invert_graph(vis, model_graph, dopsf=True, invert_single=invert_residual,
                                    iterator=vis_timeslice_iter, normalize=False, timeslice=10.0)
    
    solution_graph = create_solve_image_graph(vis, model_graph=model_graph,
                                              invert_residual=invert_residual,
                                              predict_residual=predict_residual,
                                              iterator=vis_timeslice_iter, **kwargs)
    
    residual_timeslice_graph = create_residual_graph(vis, model_graph=solution_graph,
                                                     predict_residual=predict_residual,
                                                     invert_residual=invert_residual,
                                                     iterator=vis_timeslice_iter)
    
    return create_restore_graph(solution_graph=solution_graph, psf_graph=psf_graph,
                                residual_graph=residual_timeslice_graph, **kwargs)


def rcal_dask(vis: BlockVisibility, components, **kwargs):
    """ Real-time calibration pipeline.

    Reads visibilities through a BlockVisibility iterator, calculates model visibilities according to a
    component-based sky model, and performans a calibration, writing a gaintable for each chunk of visibilities.

    :param vis: Visibility or Union(Visibility, Iterable)
    :param components: Component-based sky model
    :param kwargs: Parameters
    :returns: gaintable
   """
    
    if not isinstance(vis, collections.Iterable):
        vis = [vis]
    
    for ichunk, vischunk in enumerate(vis):
        vispred = copy_visibility(vischunk)
        vispred.data['vis'][...] = 0.0
        vispred = predict_skycomponent_blockvisibility(vispred, components)
        gt = solve_gaintable(vischunk, vispred, phase_only=False)
        yield gt


def create_calibrate_graph(vis, model_graph, **kwargs):
    """ Solve and apply calibration
    
    :param vis:
    :param model_graph:
    :param kwargs:
    :return:
    """
