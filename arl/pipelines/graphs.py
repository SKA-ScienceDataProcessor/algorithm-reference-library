""" Pipelines expressed as dask graphs
"""

from dask import delayed

from arl.data.parameters import get_parameter
from arl.image.deconvolution import restore_cube
from arl.graphs.graphs import create_deconvolve_graph, create_invert_graph, create_residual_graph, \
    create_selfcal_graph_list, create_predict_graph

def create_continuum_imaging_pipeline_graph(vis_graph_list, model_graph: delayed,
                                            c_deconvolve_graph=create_deconvolve_graph,
                                            c_invert_graph=create_invert_graph,
                                            c_predict_graph=create_predict_graph,
                                            c_residual_graph=create_residual_graph,
                                            **kwargs) -> delayed:
    """ Create graph for the continuum imaging pipeline.
    
    Same as ICAL but with no selfcal.
    
    :param vis_graph_list:
    :param model_graph:
    :param c_deconvolve_graph: Default: create_deconvolve_graph
    :param c_invert_graph: Default: create_invert_graph
    :param c_residual_graph: Default: Default: create_residual graph
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    return create_ical_pipeline_graph(vis_graph_list, model_graph,
                                      c_deconvolve_graph=c_deconvolve_graph,
                                      c_invert_graph=c_invert_graph,
                                      c_predict_graph=c_predict_graph,
                                      c_residual_graph=c_residual_graph,
                                      first_selfcal=None,
                                      **kwargs)


def create_spectral_line_imaging_pipeline_graph(vis_graph_list, model_graph: delayed,
                                                continuum_model_graph=None,
                                                c_deconvolve_graph=create_deconvolve_graph,
                                                c_invert_graph=create_invert_graph,
                                                c_predict_graph=create_predict_graph,
                                                c_residual_graph=create_residual_graph,
                                                **kwargs) -> delayed:
    """Create graph for spectral line imaging pipeline

    Uses the ical pipeline after subtraction of a continuum model
    
    :param vis_graph_list: List of visibility graphs
    :param model_graph: Spectral line model graph
    :param continuum_model_graph: Continuum model graph
    :param c_deconvolve_graph: Default: create_deconvolve_graph
    :param c_invert_graph: Default: create_invert_graph,
    :param c_residual_graph: Default: Default: create_residual graph
    :param kwargs: Parameters for functions in graphs
    :return: graphs of (deconvolved model, residual, restored)
    """
    if continuum_model_graph is not None:
        vis_graph_list = c_predict_graph(vis_graph_list, continuum_model_graph, **kwargs)
    
    return create_ical_pipeline_graph(vis_graph_list, model_graph,
                                      c_deconvolve_graph=c_deconvolve_graph,
                                      c_predict_graph=c_predict_graph,
                                      c_invert_graph=c_invert_graph,
                                      c_residual_graph=c_residual_graph,
                                      first_selfcal=None,
                                      **kwargs)


def create_ical_pipeline_graph(vis_graph_list, model_graph: delayed,
                               c_deconvolve_graph=create_deconvolve_graph,
                               c_invert_graph=create_invert_graph,
                               c_residual_graph=create_residual_graph,
                               c_predict_graph=create_predict_graph,
                               c_selfcal_graph=create_selfcal_graph_list,
                               first_selfcal=None, **kwargs) -> delayed:
    """Create graph for ICAL pipeline
    
    :param vis_graph_list:
    :param model_graph:
    :param c_deconvolve_graph: Default: create_deconvolve_graph
    :param c_invert_graph: Default: create_invert_graph,
    :param c_residual_graph: Default: Default: create_residual_graph
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    psf_graph = c_invert_graph(vis_graph_list, model_graph, dopsf=True, **kwargs)
    
    if first_selfcal is not None and first_selfcal == 0:
        vis_graph_list = c_selfcal_graph(vis_graph_list, model_graph, c_predict_graph, **kwargs)
    residual_graph = c_residual_graph(vis_graph_list, model_graph, **kwargs)
    deconvolve_model_graph = c_deconvolve_graph(residual_graph, psf_graph, model_graph, **kwargs)
    
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):
            if first_selfcal is not None and cycle >= first_selfcal:
                vis_graph_list = c_selfcal_graph(vis_graph_list, deconvolve_model_graph,
                                                 c_predict_graph, **kwargs)
            residual_graph = c_residual_graph(vis_graph_list, deconvolve_model_graph, **kwargs)
            
            deconvolve_model_graph = c_deconvolve_graph(residual_graph, psf_graph,
                                                        deconvolve_model_graph, **kwargs)
        residual_graph = c_residual_graph(vis_graph_list, deconvolve_model_graph, **kwargs)
    
    restore_graph = delayed(restore_cube, pure=True, nout=1)(deconvolve_model_graph,
                                                             psf_graph[0], residual_graph[0],
                                                             **kwargs)
    return delayed((deconvolve_model_graph, residual_graph, restore_graph))
