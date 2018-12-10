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
from wrappers.serial.griddata.gridding import grid_weight_to_griddata, griddata_reweight, griddata_merge_weights
from wrappers.serial.griddata.kernels import create_pswf_convolutionfunction
from wrappers.serial.griddata.operations import create_griddata_from_image
from wrappers.serial.image.deconvolution import deconvolve_cube, restore_cube
from wrappers.serial.image.gather_scatter import image_scatter_facets, image_gather_facets, \
    image_scatter_channels, image_gather_channels
from wrappers.serial.image.operations import calculate_image_frequency_moments
from wrappers.serial.visibility.base import copy_visibility
from wrappers.serial.visibility.gather_scatter import visibility_scatter, visibility_gather
from wrappers.serial.imaging.weighting import taper_visibility_gaussian

log = logging.getLogger(__name__)


def predict_list_serial_workflow(vis_list, model_imagelist, context, vis_slices=1, facets=1,
                                 gcfcf=None, **kwargs):
    """Predict, iterating over both the scattered vis_list and image

    The visibility and image are scattered, the visibility is predicted on each part, and then the
    parts are assembled.

    :param vis_list:
    :param model_imagelist: Model used to determine image parameters
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param facets: Number of facets (per axis)
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists
   """
    
    assert len(vis_list) == len(model_imagelist), "Model must be the same length as the vis_list"
    
    # Predict_2d does not clear the vis so we have to do it here.
    vis_list = zero_list_serial_workflow(vis_list)

    c = imaging_context(context)
    vis_iter = c['vis_iterator']
    predict = c['predict']
    
    if facets % 2 == 0 or facets == 1:
        actual_number_facets = facets
    else:
        actual_number_facets = facets - 1
    
    def predict_ignore_none(vis, model, g):
        if vis is not None:
            assert isinstance(vis, Visibility), vis
            assert isinstance(model, Image), model
            return predict(vis, model, context=context, gcfcf=g, **kwargs)
        else:
            return None
    
    if gcfcf is None:
        gcfcf = [create_pswf_convolutionfunction(m) for m in model_imagelist]
    
    # Loop over all frequency windows
    if facets == 1:
        image_results_list = list()
        for freqwin, vis_list in enumerate(vis_list):
            if len(gcfcf) > 1:
                g = gcfcf[freqwin]
            else:
                g = gcfcf[0]
            # Create the graph to divide an image into facets. This is by reference.
            # Create the graph to divide the visibility into slices. This is by copy.
            sub_vis_lists = visibility_scatter(vis_list, vis_iter, vis_slices)
            
            image_vis_lists = list()
            # Loop over sub visibility
            for sub_vis_list in sub_vis_lists:
                # Predict visibility for this sub-visibility from this image
                image_vis_list = predict_ignore_none(sub_vis_list, model_imagelist[freqwin], g)
                # Sum all sub-visibilities
                image_vis_lists.append(image_vis_list)
            image_results_list.append(visibility_gather(image_vis_lists, vis_list, vis_iter))
        
        return image_results_list
    else:
        image_results_list_list = list()
        for freqwin, vis_list in enumerate(vis_list):
            # Create the graph to divide an image into facets. This is by reference.
            facet_lists = image_scatter_facets(model_imagelist[freqwin], facets=facets)
            facet_vis_lists = list()
            sub_vis_lists = visibility_scatter(vis_list, vis_iter, vis_slices)
            
            # Loop over sub visibility
            for sub_vis_list in sub_vis_lists:
                facet_vis_results = list()
                # Loop over facets
                for facet_list in facet_lists:
                    # Predict visibility for this subvisibility from this facet
                    facet_vis_list = predict_ignore_none(sub_vis_list, facet_list,
                                                         None)
                    facet_vis_results.append(facet_vis_list)
                # Sum the current sub-visibility over all facets
                facet_vis_lists.append(sum_predict_results(facet_vis_results))
            # Sum all sub-visibilties
            image_results_list_list.append(visibility_gather(facet_vis_lists, vis_list, vis_iter))
        return image_results_list_list


def invert_list_serial_workflow(vis_list, template_model_imagelist, dopsf=False, normalize=True,
                                facets=1, vis_slices=1, context='2d', gcfcf=None, **kwargs):
    """ Sum results from invert, iterating over the scattered image and vis_list

    :param vis_list:
    :param template_model_imagelist: Model used to determine image parameters
    :param dopsf: Make the PSF instead of the dirty image
    :param facets: Number of facets
    :param normalize: Normalize by sumwt
    :param vis_slices: Number of slices
    :param context: Imaging context
    :param gcfcg: tuple containing grid correction and convolution function
    :param kwargs: Parameters for functions in components
    :return: List of (image, sumwt) tuple
   """
    
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
    
    def invert_ignore_none(vis, model, g):
        if vis is not None:
            
            return invert(vis, model, context=context, dopsf=dopsf, normalize=normalize,
                          gcfcf=g, **kwargs)
        else:
            return create_empty_image_like(model), 0.0
    
    # If we are doing facets, we need to create the gcf for each image
    if gcfcf is None and facets == 1:
        gcfcf = [create_pswf_convolutionfunction(template_model_imagelist[0])]
    
    # Loop over all vis_lists independently
    results_vislist = list()
    if facets == 1:
        for freqwin, vis_list in enumerate(vis_list):
            if len(gcfcf) > 1:
                g = gcfcf[freqwin]
            else:
                g = gcfcf[0]
            # Create the graph to divide the visibility into slices. This is by copy.
            sub_vis_lists = visibility_scatter(vis_list, vis_iter, vis_slices=vis_slices)
            
            # Iterate within each vis_list
            vis_results = list()
            for sub_vis_list in sub_vis_lists:
                vis_results.append(invert_ignore_none(sub_vis_list, template_model_imagelist[freqwin],
                                                      g))
            results_vislist.append(sum_invert_results(vis_results))
        return results_vislist
    else:
        for freqwin, vis_list in enumerate(vis_list):
            # Create the graph to divide an image into facets. This is by reference.
            facet_lists = image_scatter_facets(template_model_imagelist[
                                                   freqwin],
                                               facets=facets)
            # Create the graph to divide the visibility into slices. This is by copy.
            sub_vis_lists = visibility_scatter(vis_list, vis_iter, vis_slices=vis_slices)
            
            # Iterate within each vis_list
            vis_results = list()
            for sub_vis_list in sub_vis_lists:
                facet_vis_results = list()
                for facet_list in facet_lists:
                    facet_vis_results.append(invert_ignore_none(sub_vis_list, facet_list,
                                                                None))
                vis_results.append(gather_image_iteration_results(facet_vis_results,
                                                                  template_model_imagelist[freqwin]))
            results_vislist.append(sum_invert_results(vis_results))
    
    return results_vislist


def residual_list_serial_workflow(vis, model_imagelist, context='2d', gcfcf=None, **kwargs):
    """ Create a graph to calculate residual image using w stacking and faceting

    :param vis:
    :param model_imagelist: Model used to determine image parameters
    :param vis:
    :param model_imagelist: Model used to determine image parameters
    :param context:
    :param gcfcg: tuple containing grid correction and convolution function
    :param kwargs: Parameters for functions in components
    :return:
    """
    model_vis = zero_list_serial_workflow(vis)
    model_vis = predict_list_serial_workflow(model_vis, model_imagelist, context=context, gcfcf=gcfcf, **kwargs)
    residual_vis = subtract_list_serial_workflow(vis, model_vis)
    return invert_list_serial_workflow(residual_vis, model_imagelist, dopsf=False, normalize=True, context=context,
                                       gcfcf=gcfcf, **kwargs)


def restore_list_serial_workflow(model_imagelist, psf_imagelist, residual_imagelist=None, **kwargs):
    """ Create a graph to calculate the restored image

    :param model_imagelist: Model list
    :param psf_imagelist: PSF list
    :param residual_imagelist: Residual list
    :param kwargs: Parameters for functions in components
    :return:
    """
    
    if residual_imagelist is None:
        residual_imagelist = []
    
    if len(residual_imagelist) > 0:
        return [restore_cube(model_imagelist[i], psf_imagelist[i][0],
                             residual_imagelist[i][0], **kwargs)
                for i, _ in enumerate(model_imagelist)]
    else:
        return [restore_cube(model_imagelist[i], psf_imagelist[i][0], **kwargs)
                for i, _ in enumerate(model_imagelist)]


def deconvolve_list_serial_workflow(dirty_list, psf_list, model_imagelist, prefix='', **kwargs):
    """Create a graph for deconvolution, adding to the model

    :param dirty_list:
    :param psf_list:
    :param model_imagelist:
    :param kwargs: Parameters for functions in components
    :return: (graph for the deconvolution, graph for the flat)
    """
    nchan = len(dirty_list)
    nmoment = get_parameter(kwargs, "nmoment", 0)
    
    def deconvolve(dirty, psf, model, facet, gthreshold):
        if prefix == '':
            lprefix = "facet %d" % facet
        else:
            lprefix = "%s, facet %d" % (prefix, facet)
        
        if nmoment > 0:
            moment0 = calculate_image_frequency_moments(dirty)
            this_peak = numpy.max(numpy.abs(moment0.data[0, ...])) / dirty.data.shape[0]
        else:
            this_peak = numpy.max(numpy.abs(dirty.data[0, ...]))
        
        if this_peak > 1.1 * gthreshold:
            kwargs['threshold'] = gthreshold
            result, _ = deconvolve_cube(dirty, psf, prefix=lprefix, **kwargs)
            
            if result.data.shape[0] == model.data.shape[0]:
                result.data += model.data
            return result
        else:
            
            return copy_image(model)
    
    deconvolve_facets = get_parameter(kwargs, 'deconvolve_facets', 1)
    deconvolve_overlap = get_parameter(kwargs, 'deconvolve_overlap', 0)
    deconvolve_taper = get_parameter(kwargs, 'deconvolve_taper', None)
    if deconvolve_overlap > 0:
        deconvolve_number_facets = (deconvolve_facets - 2) ** 2
    else:
        deconvolve_number_facets = deconvolve_facets ** 2
    
    model_imagelist = image_gather_channels(model_imagelist)
    
    # Scatter the separate channel images into deconvolve facets and then gather channels for each facet.
    # This avoids constructing the entire spectral cube.
    #    dirty_list = remove_sumwt, nout=nchan)(dirty_list)
    scattered_channels_facets_dirty_list = \
        [image_scatter_facets(d[0], facets=deconvolve_facets,
                              overlap=deconvolve_overlap,
                              taper=deconvolve_taper)
         for d in dirty_list]
    
    # Now we do a transpose and gather
    scattered_facets_list = [
        image_gather_channels([scattered_channels_facets_dirty_list[chan][facet]
                               for chan in range(nchan)])
        for facet in range(deconvolve_number_facets)]
    
    psf_list = remove_sumwt(psf_list)
    psf_list = image_gather_channels(psf_list)
    
    scattered_model_imagelist = \
        image_scatter_facets(model_imagelist,
                             facets=deconvolve_facets,
                             overlap=deconvolve_overlap)
    
    # Work out the threshold. Need to find global peak over all dirty_list images
    threshold = get_parameter(kwargs, "threshold", 0.0)
    fractional_threshold = get_parameter(kwargs, "fractional_threshold", 0.1)
    nmoment = get_parameter(kwargs, "nmoment", 0)
    use_moment0 = nmoment > 0
    
    # Find the global threshold. This uses the peak in the average on the frequency axis since we
    # want to use it in a stopping criterion in a moment clean
    global_threshold = threshold_list(scattered_facets_list, threshold, fractional_threshold, use_moment0=use_moment0,
                                      prefix=prefix)
    
    facet_list = numpy.arange(deconvolve_number_facets).astype('int')
    scattered_results_list = [
        deconvolve(d, psf_list, m, facet, global_threshold)
        for d, m, facet in zip(scattered_facets_list, scattered_model_imagelist, facet_list)]
    
    # Gather the results back into one image, correcting for overlaps as necessary. The taper function is is used to
    # feather the facets together
    gathered_results_list = image_gather_facets(scattered_results_list, model_imagelist,
                                                facets=deconvolve_facets,
                                                overlap=deconvolve_overlap,
                                                taper=deconvolve_taper)
    flat_list = image_gather_facets(scattered_results_list, model_imagelist,
                                    facets=deconvolve_facets, overlap=deconvolve_overlap,
                                    taper=deconvolve_taper, return_flat=True)
    
    return image_scatter_channels(gathered_results_list, subimages=nchan), flat_list


def deconvolve_channel_list_serial_workflow(dirty_list, psf_list, model_imagelist, subimages, **kwargs):
    """Create a graph for deconvolution by channels, adding to the model

    Does deconvolution channel by channel.
    :param subimages:
    :param dirty_list:
    :param psf_list: Must be the size of a facet
    :param model_imagelist: Current model
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
    
    output = create_empty_image_like(model_imagelist)
    dirty_lists = image_scatter_channels(dirty_list[0],
                                         subimages=subimages)
    results = [deconvolve_subimage(dirty_list, psf_list[0])
               for dirty_list in dirty_lists]
    result = image_gather_channels(results, output, subimages=subimages)
    return add_model(result, model_imagelist)


def weight_list_serial_workflow(vis_list, model_imagelist, gcfcf=None, weighting='uniform', **kwargs):
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
    
    if gcfcf is None:
        gcfcf = [create_pswf_convolutionfunction(model_imagelist[centre])]
    
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
    
    weight_list = [grid_wt(vis_list[i], model_imagelist[i], gcfcf) for i in range(len(vis_list))]
    
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
    
    return [re_weight(v, model_imagelist[i], merged_weight_grid, gcfcf)
            for i, v in enumerate(vis_list)]
    
def taper_list_serial_workflow(vis_list, size_required):
    """Taper to desired size
    
    :param vis_list:
    :param size_required:
    :return:
    """
    return [taper_visibility_gaussian(v, beam=size_required) for v in vis_list]


def zero_list_serial_workflow(vis_list):
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
    
    return [zero(v) for v in vis_list]


def subtract_list_serial_workflow(vis_list, model_vislist):
    """ Initialise vis to zero

    :param vis_list:
    :param model_vislist: Model to be subtracted
    :return: List of vis_lists
   """
    
    def subtract_vis(vis, model_vis):
        if vis is not None and model_vis is not None:
            assert vis.vis.shape == model_vis.vis.shape
            subvis = copy_visibility(vis)
            subvis.data['vis'][...] -= model_vis.data['vis'][...]
            return subvis
        else:
            return None
    
    return [subtract_vis(vis=vis_list[i], model_vis=model_vislist[i]) for i in range(len(vis_list))]
