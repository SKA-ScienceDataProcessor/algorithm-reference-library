"""Manages the imaging context in MPI. This take a string and returns a dictionary containing:
 * Predict function
 * Invert function
 * image_iterator function
 * vis_iterator function
 Parallel version of workflows.serial.imaging.inaging_serial.py

"""

import collections
import logging

import numpy

from data_models.memory_data_models import Image
from data_models.parameters import get_parameter
from processing_library.image.operations import copy_image, create_empty_image_like
from processing_components.image.deconvolution import deconvolve_cube, restore_cube
from processing_components.image.gather_scatter import image_scatter_facets, image_gather_facets, \
    image_scatter_channels, image_gather_channels
from processing_components.image.operations import calculate_image_frequency_moments
from processing_components.imaging.weighting import weight_visibility
from processing_components.visibility.base import copy_visibility
from processing_components.visibility.gather_scatter import visibility_scatter, visibility_gather
from workflows.shared.imaging.imaging_shared import imaging_context
from workflows.shared.imaging.imaging_shared import sum_invert_results, remove_sumwt, sum_predict_results, \
    threshold_list

from mpi4py import MPI

def predict_list_mpi_workflow(vis_list, model_imagelist, vis_slices=1,
                                 facets=1, context='2d', comm=MPI.COMM_WORLD, **kwargs):
    """Predict, iterating over both the scattered vis_list and image

    The visibility and image are scattered, the visibility is predicted on each part, and then the
    parts are assembled.

    :param vis_list:
    :param model_imagelist: Model used to determine image parameters
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param facets: Number of facets (per axis)
    :param context:
    :param comm: MPI communicator
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists
   """
    rank = comm.Get_rank()
    size = comm.Get_size()
    # the assert only makes sense in proc 0 as for the others both lists are
    # empty
    assert len(vis_list) == len(model_imagelist), "Model must be the same length as the vis_list"

    c = imaging_context(context)
    vis_iter = c['vis_iterator']
    predict = c['predict']

    def predict_ignore_none(vis, model):
        if vis is not None:
            return predict(vis, model, context=context, facets=facets, vis_slices=vis_slices)
        else:
            return None
    
    vis_list_len=comm.bcast(len(vis_list),root=0)
    print('%d: %d (%d) elements in vis_list' % (rank,len(vis_list),vis_list_len))
    print(vis_list)

    image_results_list_list = list()
    #NOTE: We could parallelize here by freqwin instead of inside that would
    # reduce data transfers
    # Loop over all frequency windows
    # for i in range(vis_list_len):
    if rank == 0:
        for freqwin, vis_lst in enumerate(vis_list):
            print('%d: freqwin %d vis_lst:' %(rank,freqwin))
            print(vis_lst)
            # Create the graph to divide an image into facets. This is by reference.
            facet_lists = image_scatter_facets(model_imagelist[freqwin], facets=facets)
            # facet_lists = numpy.array_split(facet_lists, size)
            # Create the graph to divide the visibility into slices. This is by copy.
            sub_vis_lists = visibility_scatter(vis_lst, vis_iter, vis_slices)
            #print('%d: sub_vis_list after visibility_scatter in %d vis_slices'
            #      %(rank,vis_slices))
            #print(sub_vis_lists)
            sub_vis_lists = numpy.array_split(sub_vis_lists, size)
            ## Scater facets and visibility lists to all processes
            facet_lists=comm.bcast(facet_lists,root=0)
            sub_sub_vis_lists=comm.scatter(sub_vis_lists,root=0)

            ## All processes compute its part
            facet_vis_lists = list()
            # Loop over sub visibility
            for sub_vis_list in sub_sub_vis_lists:
                facet_vis_results = list()
                # Loop over facets
                for facet_list in facet_lists:
                    # Predict visibility for this subvisibility from this facet
                    facet_vis_list = predict_ignore_none(sub_vis_list, facet_list)
                    facet_vis_results.append(facet_vis_list)
                # Sum the current sub-visibility over all facets
                facet_vis_lists.append(sum_predict_results(facet_vis_results))
            ## gather results from all processes
            facet_vis_lists=comm.gather(facet_vis_lists,root=0)
            # Sum all sub-visibilties
            facet_vis_lists=numpy.concatenate(facet_vis_lists)
            image_results_list_list.append(visibility_gather(facet_vis_lists,
                                                             vis_lst,
                                                             vis_iter))

    else:
        for i in range(vis_list_len):
        #for freqwin, vis_lst in enumerate(vis_list):
            print('%d: iteration %d' %(rank,i))
            facet_lists = list()
            sub_vis_lists = list()
            ## Scater facets and visibility lists to all processes
            facet_lists =comm.bcast(facet_lists,root=0)
            sub_sub_vis_lists=comm.scatter(sub_vis_lists,root=0)
            #print('%d sub_sub_vis_list' % rank)
            #print(sub_sub_vis_lists)
            #print('%d facet_lists' % rank)
            #print(facet_lists)
            ## All processes compute its part
            facet_vis_lists = list()
            # Loop over sub visibility
            for sub_vis_list in sub_sub_vis_lists:
                facet_vis_results = list()
                # Loop over facets
                for facet_list in facet_lists:
                    # Predict visibility for this subvisibility from this facet
                    facet_vis_list = predict_ignore_none(sub_vis_list, facet_list)
                    facet_vis_results.append(facet_vis_list)
                # Sum the current sub-visibility over all facets
                facet_vis_lists.append(sum_predict_results(facet_vis_results))
            ## gather results from all processes
            facet_vis_lists=comm.gather(facet_vis_lists,root=0)
            image_results_list_list=list()

    return image_results_list_list


def invert_list_mpi_workflow(vis_list, template_model_imagelist, dopsf=False, normalize=True,
                                facets=1, vis_slices=1, context='2d',
                                comm=MPI.COMM_WORLD, **kwargs):
    """ Sum results from invert, iterating over the scattered image and vis_list

    :param vis_list:
    :param template_model_imagelist: Model used to determine image parameters
    :param dopsf: Make the PSF instead of the dirty image
    :param facets: Number of facets
    :param normalize: Normalize by sumwt
    :param vis_slices: Number of slices
    :param context: Imaging context
    :param comm: MPI Communicator
    :param kwargs: Parameters for functions in components
    :return: List of (image, sumwt) tuple
   """
   
   # NOTE: Be careful with normalization as normalizing parts is not the 
   # same as normalizing the whole

    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank==0:
        if not isinstance(template_model_imagelist, collections.Iterable):
            template_model_imagelist = [template_model_imagelist]
    
    c = imaging_context(context)
    vis_iter = c['vis_iterator']
    invert = c['invert']
    
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
    
    vis_list_len=comm.bcast(len(vis_list),root=0)
    print('%d: %d (%d) elements in vis_list' % (rank,len(vis_list),vis_list_len))
    print(vis_list)
    results_vislist = list()
    #NOTE: We could parallelize here by freqwin instead of inside that would
    # reduce data transfers
    # Loop over all vis_lists independently
    if rank == 0:
        for freqwin, vis_list in enumerate(vis_list):
            # Create the graph to divide an image into facets. This is by reference.
            template_model_imagelist_fwin=comm.bcast(template_model_imagelist[freqwin],root=0)
            facet_lists = image_scatter_facets(template_model_imagelist[
                                               freqwin],
                                           facets=facets)
            # Create the graph to divide the visibility into slices. This is by copy.
            sub_vis_lists = visibility_scatter(vis_list, vis_iter,
                                           vis_slices=vis_slices)
            sub_vis_lists= numpy.array_split(sub_vis_lists,size)
            sub_sub_vis_lists = comm.scatter(sub_vis_lists,root=0)
            # Iterate within each vis_list
            vis_results = list()
            for sub_vis_list in sub_sub_vis_lists:
                facet_vis_results = list()
                for facet_list in facet_lists:
                    facet_vis_results.append(invert_ignore_none(sub_vis_list, facet_list))
                vis_results.append(gather_image_iteration_results(facet_vis_results,
                                                              template_model_imagelist[freqwin]))
            all_vis_results=comm.gather(vis_results, root=0)
            all_vis_results=numpy.concatenate(all_vis_results)
            # sum_invert_results normalized according to weigths it must be
            # done to the full set of visibilities
            results_vislist.append(sum_invert_results(all_vis_results))
    else:
        for i in range(vis_list_len):
        #for freqwin, vis_lst in enumerate(vis_list):
            print('%d: iteration %d' %(rank,i))
            template_model_imagelist_fwin=list()
            template_model_imagelist_fwin=comm.bcast(template_model_imagelist_fwin,root=0)
            facet_lists = image_scatter_facets(template_model_imagelist_fwin,
                                           facets=facets)
            sub_vis_lists = list()
            facet_lists=comm.bcast(facet_lists,root=0)
            sub_sub_vis_lists = comm.scatter(sub_vis_lists,root=0)
            # Iterate within each vis_list
            vis_results = list()
            for sub_vis_list in sub_sub_vis_lists:
                facet_vis_results = list()
                for facet_list in facet_lists:
                    facet_vis_results.append(invert_ignore_none(sub_vis_list, facet_list))
                vis_results.append(gather_image_iteration_results(facet_vis_results,
                                                              template_model_imagelist_fwin))
            all_vis_results=comm.gather(vis_results, root=0)
            results_vislist=list()

    return results_vislist


