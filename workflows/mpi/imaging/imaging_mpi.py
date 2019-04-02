"""Manages the imaging context. This take a string and returns a dictionary containing:
 * Predict function
 * Invert function
 * image_iterator function
 * vis_iterator function

"""

import collections
import logging

import numpy

from data_models.memory_data_models import Image, Visibility
from data_models.parameters import get_parameter
from processing_library.image.operations import copy_image, create_empty_image_like
from workflows.shared.imaging.imaging_shared import imaging_context
from workflows.shared.imaging.imaging_shared import sum_invert_results, remove_sumwt, sum_predict_results, \
    threshold_list
from wrappers.mpi.griddata.gridding import grid_weight_to_griddata, griddata_reweight, griddata_merge_weights
from wrappers.mpi.griddata.kernels import create_pswf_convolutionfunction
from wrappers.mpi.griddata.operations import create_griddata_from_image
from wrappers.mpi.image.deconvolution import deconvolve_cube, restore_cube
from wrappers.mpi.image.gather_scatter import image_scatter_facets, image_gather_facets, \
    image_scatter_channels, image_gather_channels
from wrappers.mpi.image.operations import calculate_image_frequency_moments
from wrappers.mpi.visibility.base import copy_visibility
from wrappers.mpi.visibility.gather_scatter import visibility_scatter, visibility_gather
from wrappers.mpi.imaging.weighting import taper_visibility_gaussian, taper_visibility_tukey

from mpi4py import MPI
import sys

log = logging.getLogger(__name__)

def predict_list_mpi_workflow(vis_list, model_imagelist, context, vis_slices=1, facets=1,
                                     gcfcf=None, comm=MPI.COMM_WORLD, **kwargs):
    """Predict, iterating over both the scattered vis_list and image
    
    The visibility and image are scattered, the visibility is predicted on each part, and then the
    parts are assembled. About data distribution: vis_list and model_imagelist
    live in rank 0; vis_slices, facets, context are replicated in all nodes.
    gcfcf if exists lives in rank 0; if not every mpi proc will create its own
    for the  corresponding subset of the image.

    :param vis_list:
    :param model_imagelist: Model used to determine image parameters
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param facets: Number of facets (per axis)
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param comm: MPI communicator
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists
   """
    rank = comm.Get_rank()
    size = comm.Get_size()
    log.info('%d: In predict_list_mpi_workflow: %d elements in vis_list' % (rank,len(vis_list)))
    # the assert only makes sense in proc 0 as for the others both lists are
    # empty
    assert len(vis_list) == len(model_imagelist), "Model must be the same length as the vis_list"

    # The use_serial_predict version paralelizes by freq (my opt version) 
    assert get_parameter(kwargs, "use_serial_predict", True),"Only freq paralellization implemented"
    #if get_parameter(kwargs, "use_serial_predict", False):
    if get_parameter(kwargs, "use_serial_predict", True):
        from workflows.serial.imaging.imaging_serial import predict_list_serial_workflow
        
        image_results_list = list()
        # Distribute visibilities and model by freq
        sub_vis_list= numpy.array_split(vis_list, size)
        sub_vis_list=comm.scatter(sub_vis_list,root=0)
        sub_model_imagelist= numpy.array_split(model_imagelist, size)
        sub_model_imagelist=comm.scatter(sub_model_imagelist,root=0)
        if gcfcf is not None:
            sub_gcfcf = numpy.array_split(gcfcf,size)
            sub_gcfcf=comm.scatter(sub_gcfcf,root=0)
        isinstance(sub_vis_list[0], Visibility)
        image_results_list= [predict_list_serial_workflow(vis_list=[sub_vis_list[i]],
                     model_imagelist=[sub_model_imagelist[i]], vis_slices=vis_slices,
                     facets=facets, context=context, gcfcf=gcfcf, **kwargs)[0]
                for i, _ in enumerate(sub_vis_list)]
        #print(image_results_list)
        
        image_results_list=comm.gather(image_results_list,root=0)
        if rank == 0:
            #image_results_list_list=[x for x in image_results_list_list if x]
            image_results_list=numpy.concatenate(image_results_list)
        else:
            image_results_list=list()

    return image_results_list



def invert_list_mpi_workflow(vis_list, template_model_imagelist, context, dopsf=False, normalize=True,
                                    facets=1, vis_slices=1, gcfcf=None, 
                                comm=MPI.COMM_WORLD, **kwargs):
    """ Sum results from invert, iterating over the scattered image and vis_list

    :param vis_list: Only full for rank==0
    :param template_model_imagelist: Model used to determine image parameters
    (in rank=0)
    :param dopsf: Make the PSF instead of the dirty image
    :param facets: Number of facets
    :param normalize: Normalize by sumwt
    :param vis_slices: Number of slices
    :param context: Imaging context
    :param gcfcg: tuple containing grid correction and convolution function (in
    rank=0)
    :param comm:MPI Communicator
    :param kwargs: Parameters for functions in components
    :return: List of (image, sumwt) tuple
   """
    
   # NOTE: Be careful with normalization as normalizing parts is not the 
   # same as normalizing the whole, normalization happens for each image in a
   # frequency window (in this versio we only parallelize at freqwindows
    def concat_tuples(list_of_tuples):
        if len(list_of_tuples)<2:
            result_list=list_of_tuples
        else:
            result_list=list_of_tuples[0]
            for l in list_of_tuples[1:]:
                result_list+=l
        return result_list


    rank = comm.Get_rank()
    size = comm.Get_size()
    log.info('%d: In invert_list_mpi_workflow: %d elements in vis_list %d in model' %
          (rank,len(vis_list),len(template_model_imagelist)))

    assert get_parameter(kwargs, "use_serial_invert", True),"Only freq paralellization implemented"
    #if get_parameter(kwargs, "use_serial_invert", False):
    if get_parameter(kwargs, "use_serial_invert", True):
        from workflows.serial.imaging.imaging_serial import invert_list_serial_workflow

        results_vislist = list()
        # Distribute visibilities and model by freq
        sub_vis_list= numpy.array_split(vis_list, size)
        sub_vis_list=comm.scatter(sub_vis_list,root=0)
        sub_template_model_imagelist= numpy.array_split(template_model_imagelist, size)
        sub_template_model_imagelist=comm.scatter(sub_template_model_imagelist,root=0)
        if gcfcf is not None:
            sub_gcfcf = numpy.array_split(gcfcf,size)
            sub_gcfcf=comm.scatter(sub_gcfcf,root=0)
        isinstance(sub_vis_list[0], Visibility)
        sub_results_vislist = [invert_list_serial_workflow(vis_list=[sub_vis_list[i]],
                     template_model_imagelist=[sub_template_model_imagelist[i]],
                     context=context, dopsf=dopsf, normalize=normalize, 
                                                       vis_slices=vis_slices,
                     facets=facets, gcfcf=gcfcf, **kwargs)[0]
                for i, _ in enumerate(sub_vis_list)]
        #print("%d sub_results_vislist" %rank,sub_results_vislist)
        results_vislist=comm.gather(sub_results_vislist,root=0)
        #print("%d results_vislist before concatenate"%rank,results_vislist)
        if rank == 0:
            #image_results_list_list=[x for x in image_results_list_list if x]
            #results_vislist=numpy.concatenate(results_vislist)
            # TODO: concatenate dos not concatenate well a list of tuples 
            # it returns a 2d array instead of a concatenated list of tuples

            results_vislist =concat_tuples(results_vislist)
        else:
            results_vislist=list()
    #print("%d results_vislist"%rank,results_vislist)
    return results_vislist



def residual_list_mpi_workflow(vis, model_imagelist, context='2d', gcfcf=None,comm=MPI.COMM_WORLD, **kwargs):
    """ Create a graph to calculate residual image using w stacking and faceting

    :param context:
    :param vis: rank0
    :param model_imagelist: Model used to determine image parameters rank0
    :param gcfcg: tuple containing grid correction and convolution function
    :param kwargs: Parameters for functions in components
    :return:
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    model_vis = zero_list_mpi_workflow(vis)
    log.info('%d: In residual_list_mpi_workflow vis len %d model_imagelist len %d model_vis len %d'
          %(rank,len(vis),len(model_imagelist),len(model_vis)))

    model_vis = predict_list_mpi_workflow(model_vis, model_imagelist,
                                          context=context,gcfcf=gcfcf, **kwargs)
    residual_vis = subtract_list_mpi_workflow(vis, model_vis)
    return invert_list_mpi_workflow(residual_vis, model_imagelist, dopsf=False,
                                    normalize=True,
                                    context=context,gcfcf=gcfcf,
                                       **kwargs)


def restore_list_mpi_workflow(model_imagelist, psf_imagelist,
                              residual_imagelist,comm=MPI.COMM_WORLD, **kwargs):
    """ Create a graph to calculate the restored image

    :param model_imagelist: Model list (rank0)
    :param psf_imagelist: PSF list (rank0)
    :param residual_imagelist: Residual list (rank0)
    :param kwargs: Parameters for functions in components
    :return:
    """
    from workflows.serial.imaging.imaging_serial import restore_list_serial_workflow_nosumwt
    rank = comm.Get_rank()
    size = comm.Get_size()
    #TODO Parallelize! and check the dask version, it removes sumwt component
    # to reduce communication
    if residual_imagelist is None:
        residual_imagelist = []
    
    if rank==0:
        psf_list = remove_sumwt(psf_imagelist)
        if len(residual_imagelist) > 0:
            residual_list = remove_sumwt(residual_imagelist)
        else:
            residual_list = residual_imagelist
    else:
        psf_list=list()
        residual_list=list()

    sub_model_imagelist=numpy.array_split(model_imagelist,size)
    sub_model_imagelist=comm.scatter(sub_model_imagelist,root=0)
    sub_psf_list=numpy.array_split(psf_list,size)
    sub_psf_list=comm.scatter(sub_psf_list,root=0)
    sub_residual_list=numpy.array_split(residual_list,size)
    sub_residual_list=comm.scatter(sub_residual_list,root=0)

    sub_result_list=restore_list_serial_workflow_nosumwt(sub_model_imagelist,
                                         sub_psf_list,
                                         sub_residual_list)
    #sub_result_list=[restore_cube(sub_model_imagelist[i], sub_psf_list[i],
    #                     sub_residual_list[i], **kwargs)
    #            for i, _ in enumerate(sub_model_imagelist)]
    result_list=comm.gather(sub_result_list,root=0)
    if rank==0:
        # this is a list of tuples too, we may need to call my function
        result_list=numpy.concatenate(result_list)
    else:
        result_list=list()
    return result_list



def deconvolve_list_mpi_workflow(dirty_list, psf_list, model_imagelist,
                                        prefix='',
                                        mask=None,comm=MPI.COMM_WORLD, **kwargs):
    """Create a graph for deconvolution, adding to the model

    :param dirty_list: in rank0
    :param psf_list: in rank0
    :param model_imagelist: in rank0
    :param prefix: Informative prefix to log messages
    :param mask: Mask for deconvolution
    :param comm: MPI communicator
    :param kwargs: Parameters for functions in components
    :return: graph for the deconvolution
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    nchan = len(dirty_list)
    log.info('%d: deconvolve_list_mpi_workflow: dirty_list len %d psf_list len %d model_imagelist len %d' %(rank,len(dirty_list),len(psf_list),len(model_imagelist)))

    nmoment = get_parameter(kwargs, "nmoment", 0)
    assert get_parameter(kwargs, "use_serial_clean", True),"Only serial deconvolution implemented"
    if get_parameter(kwargs, "use_serial_clean", True):
        from workflows.serial.imaging.imaging_serial import deconvolve_list_serial_workflow
        if rank==0:
            assert isinstance(model_imagelist, list), model_imagelist
            result_list=deconvolve_list_serial_workflow (dirty_list, psf_list, model_imagelist, prefix=prefix, mask=mask, **kwargs)
        else:
            result_list=list()
    return result_list


def deconvolve_list_channel_mpi_workflow(dirty_list, psf_list, model_imagelist,
                                         subimages, comm=MPI.COMM_WORLD,**kwargs):
    """Create a graph for deconvolution by channels, adding to the model

    Does deconvolution channel by channel.
    :param subimages: MONTSE: number of subimages (= freqchannels?)
    :param dirty_list: in rank=0
    :param psf_list: Must be the size of a facet in rank=0
    :param model_imagelist: Current model in rank=0
    :param kwargs: Parameters for functions in components
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
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank==0:
        output = create_empty_image_like(model_imagelist)
    dirty_lists = image_scatter_channels(dirty_list[0], subimages=subimages)
    sub_dirty_lists = numpy.array_split(dirty_lists,size)
    sub_dirty_lists = comm.scatter(sub_dirty_lists, root=0)
    psf_list_im=comm.Bcast(psf_list[0],root=0)
    
    sub_results = [deconvolve_subimage(dirty_list, psf_list_im)
               for dirty_list in sub_dirty_lists]
    results=comm.gather(sub_results,root=0)
    # NOTE: This is same as in invert, not scalable, we should use a reduction
    # instead but I don't understand image_gather_channels ...
    if rank==0:
        results=numpy.concatenate(results)
        result = image_gather_channels(results, output, subimages=subimages)
        result = add_model(result, model_imagelist)
    else:
        result=None
    return result



def weight_list_mpi_workflow(vis_list, model_imagelist, gcfcf=None,
                             weighting='uniform',comm=MPI.COMM_WORLD, **kwargs):
    """ Weight the visibility data
    
    This is done collectively so the weights are summed over all vis_lists and then
    corrected

    :param vis_list:
    :param model_imagelist: Model required to determine weighting parameters
    :param weighting: Type of weighting
    :param kwargs: Parameters for functions in graphs
    :return: List of vis_graphs
   """
    centre = len(model_imagelist) // 2
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if gcfcf is None:
        if rank==0:
            gcfcf = [create_pswf_convolutionfunction(model_imagelist[centre])]
    gcfcf = comm.bcast(gcfcf,root=0)

    
    def grid_wt(vis, model, g):
        if vis is not None:
            if model is not None:
                griddata = create_griddata_from_image(model)
                griddata = grid_weight_to_griddata(vis, griddata, g[0][1])
                return griddata
            else:
                return None
        else:
            return None
    
      
    sub_vis_list= numpy.array_split(vis_list, size)
    sub_vis_list=comm.scatter(sub_vis_list,root=0)
    sub_model_imagelist= numpy.array_split(model_imagelist, size)
    sub_model_imagelist=comm.scatter(sub_model_imagelist,root=0)
    
    sub_weight_list = [grid_wt(sub_vis_list[i], sub_model_imagelist[i], gcfcf)
                   for i in range(len(sub_vis_list))]
   
    weight_list=comm.allgather(sub_weight_list,root=0)
    weight_list=numpy.concatenate(weight_list)
    # This reduces a grid for each freqwin to a single grid  AllReduce but it
    # is not a simple reduction but I think it can be done with  4 reductions
    # Actually it seems I only use the  gd[0].data which would be the first
    # reduction only! TODO: try it out
    ##merged_sub_weight_grid = griddata_merge_weights(sub_weight_list)
    ##merged_weight_grid = comm.Allreduce(merged_sub_weight_grid[0].data,MPI.SUM)
    # Note that griddata_merge_weights does some averaging, if we need more
    # than the data component that would not be correct!
    merged_weight_grid = griddata_merge_weights(weight_list)
    
    def re_weight(vis, model, gd, g):
        if gd is not None:
            if vis is not None:
                # Ensure that the griddata has the right axes so that the convolution
                # function mapping works
                agd = create_griddata_from_image(model)
                agd.data = gd[0].data
                vis = griddata_reweight(vis, agd, g[0][1])
                return vis
            else:
                return None
        else:
            return vis
    
    sub_result = [re_weight(v, sub_model_imagelist[i], merged_weight_grid, gcfcf)
            for i, v in enumerate(sub_vis_list)]
    result=comm.gather(sub_result,root=0)
    if rank==0:
        result=numpy.concatenate(result)
    else:
        result=list()
    return result

def taper_list_mpi_workflow(vis_list, size_required,comm=MPI.COMM_WORLD ):
    """Taper to desired size
    
    :param vis_list:
    :param size_required:
    :return:
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    sub_vis_list= numpy.array_split(vis_list, size)
    sub_vis_list=comm.scatter(sub_vis_list,root=0)
    sub_result = [taper_visibility_gaussian(v, beam=size_required) for v in sub_vis_list]
    result=comm.gather(sub_result,root=0)
    if rank==0:
        result=numpy.concatenate(result)
    else:
        result=list()

    return result


def zero_list_mpi_workflow(vis_list,comm=MPI.COMM_WORLD ):
    """ Initialise vis to zero: creates new data holders

    :param vis_list:
    :return: List of vis_lists
   """
    
    def zero(vis):
        if vis is not None:
            zerovis = copy_visibility(vis)
            zerovis.data['vis'][...] = 0.0
            return zerovis
        else:
            return None
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    sub_vis_list= numpy.array_split(vis_list, size)
    sub_vis_list=comm.scatter(sub_vis_list,root=0)
    sub_result = [zero(v) for v in sub_vis_list]
    result=comm.gather(sub_result,root=0)
    if rank==0:
        result=numpy.concatenate(result)
    else:
        result=list()
    return result

def subtract_list_mpi_workflow(vis_list, model_vislist,comm=MPI.COMM_WORLD ):
    """ Initialise vis to zero

    :param vis_list:
    :param model_vislist: Model to be subtracted
    :return: List of vis_lists
   """
    
    def subtract_vis(vis, model_vis):
        if vis is not None and model_vis is not None:
            assert vis.vis.shape == model_vis.vis.shape,vis
            subvis = copy_visibility(vis)
            subvis.data['vis'][...] -= model_vis.data['vis'][...]
            return subvis
        else:
            return None
    rank = comm.Get_rank()
    size = comm.Get_size()
    print("%d: In substract: vis_list" %rank,vis_list)
    sub_vis_list= numpy.array_split(vis_list, size)
    sub_vis_list=comm.scatter(sub_vis_list,root=0)
    sub_model_vislist= numpy.array_split(model_vislist, size)
    sub_model_vislist=comm.scatter(sub_model_vislist,root=0)
    print("%d In substract" %rank,"model:",sub_model_vislist,"vis:",sub_vis_list)
    sub_result = [subtract_vis(vis=sub_vis_list[i], model_vis=sub_model_vislist[i])
            for i in range(len(sub_vis_list))]
    result=comm.gather(sub_result,root=0)
    if rank==0:
        result=numpy.concatenate(result)
    else:
        result=list()
    return result

