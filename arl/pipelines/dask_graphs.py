from dask import delayed

import os
import sys

from dask import delayed

import numpy

from arl.visibility.operations import create_visibility_from_rows, copy_visibility
from arl.image.deconvolution import deconvolve_cube
from arl.visibility.iterators import vis_timeslice_iter
from arl.fourier_transforms.ftprocessor import invert_2d, residual_image
from arl.data.parameters import get_parameter


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
        nresults = len(results)
        for i, result in enumerate(results):
            if i > 0:
                acc.data += result[0].data
                sumwt += result[1]
            else:
                acc = result[0]
                sumwt = result[1]
        
        if normalize:
            acc.data /= float(sumwt)
        
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
        
        acc.data /= float(sumwt)
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


def create_solve_image_graph(vis,
                             model_graph,
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