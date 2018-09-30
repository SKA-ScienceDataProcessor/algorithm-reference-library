"""Manages the imaging context. This take a string and returns a dictionary containing:
 * Predict function
 * Invert function
 * image_iterator function
 * vis_iterator function

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

log = logging.getLogger(__name__)


def predict_list_serial_workflow(vis_list, model_imagelist, vis_slices=1, facets=1, context='2d', **kwargs):
    """Predict, iterating over both the scattered vis_list and image

    The visibility and image are scattered, the visibility is predicted on each part, and then the
    parts are assembled.

    :param vis_list:
    :param model_imagelist: Model used to determine image parameters
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param facets: Number of facets (per axis)
    :param context:
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists
   """
    
    assert len(vis_list) == len(model_imagelist), "Model must be the same length as the vis_list"
    
    c = imaging_context(context)
    vis_iter = c['vis_iterator']
    predict = c['predict']
    
    def predict_ignore_none(vis, model):
        if vis is not None:
            return predict(vis, model, context=context, facets=facets, vis_slices=vis_slices, **kwargs)
        else:
            return None
    
    image_results_list_list = list()
    # Loop over all frequency windows
    if facets == 1:
        image_results_list = list()
        for freqwin, vis_list in enumerate(vis_list):
            # Create the graph to divide an image into facets. This is by reference.
            # Create the graph to divide the visibility into slices. This is by copy.
            sub_vis_lists = visibility_scatter(vis_list, vis_iter, vis_slices)
        
            image_vis_lists = list()
            # Loop over sub visibility
            for sub_vis_list in sub_vis_lists:
                # Predict visibility for this sub-visibility from this image
                image_vis_list = predict_ignore_none(sub_vis_list, model_imagelist[freqwin])
                # Sum all sub-visibilities
                image_vis_lists.append(image_vis_list)
            image_results_list.append(visibility_gather(image_vis_lists, vis_list, vis_iter))
    
        return image_results_list
    else:
        for freqwin, vis_list in enumerate(vis_list):
            # Create the graph to divide an image into facets. This is by reference.
            facet_lists = image_scatter_facets(model_imagelist[freqwin], facets=facets)
            # Create the graph to divide the visibility into slices. This is by copy.
            sub_vis_lists = visibility_scatter(vis_list, vis_iter, vis_slices)
        
            facet_vis_lists = list()
            # Loop over sub visibility
            for sub_vis_list in sub_vis_lists:
                facet_vis_results = list()
                # Loop over facets
                for facet_list in facet_lists:
                    # Predict visibility for this subvisibility from this facet
                    facet_vis_list = predict_ignore_none(sub_vis_list, facet_list)
                    facet_vis_results.append(facet_vis_list)
                # Sum the current sub-visibility over all facets
                facet_vis_lists.append(sum_predict_results(facet_vis_results))
            # Sum all sub-visibilties
            image_results_list_list.append(visibility_gather(facet_vis_lists, vis_list, vis_iter))
        return image_results_list_list


def invert_list_serial_workflow(vis_list, template_model_imagelist, dopsf=False, normalize=True,
                                facets=1, vis_slices=1, context='2d', **kwargs):
    """ Sum results from invert, iterating over the scattered image and vis_list

    :param vis_list:
    :param template_model_imagelist: Model used to determine image parameters
    :param dopsf: Make the PSF instead of the dirty image
    :param facets: Number of facets
    :param normalize: Normalize by sumwt
    :param vis_slices: Number of slices
    :param context: Imaging context
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
    
    def invert_ignore_none(vis, model):
        if vis is not None:
            return invert(vis, model, context=context, dopsf=dopsf, normalize=normalize, facets=facets,
                          vis_slices=vis_slices, **kwargs)
        else:
            return create_empty_image_like(model), 0.0
    
    # Loop over all vis_lists independently
    results_vislist = list()
    if facets == 1:
        for freqwin, vis_list in enumerate(vis_list):
            # Create the graph to divide the visibility into slices. This is by copy.
            sub_vis_lists = visibility_scatter(vis_list, vis_iter, vis_slices=vis_slices)
        
            # Iterate within each vis_list
            vis_results = list()
            for sub_vis_list in sub_vis_lists:
                vis_results.append(invert_ignore_none(sub_vis_list, template_model_imagelist[freqwin]))
            results_vislist.append(sum_invert_results(vis_results))
        return results_vislist
    else:
        for freqwin, vis_list in enumerate(vis_list):
            # Create the graph to divide an image into facets. This is by reference.
            facet_lists = image_scatter_facets(template_model_imagelist[
                                                   freqwin],
                                               facets=facets)
            # Create the graph to divide the visibility into slices. This is by copy.
            sub_vis_lists = visibility_scatter(vis_list, vis_iter,
                                               vis_slices=vis_slices)
        
            # Iterate within each vis_list
            vis_results = list()
            for sub_vis_list in sub_vis_lists:
                facet_vis_results = list()
                for facet_list in facet_lists:
                    facet_vis_results.append(invert_ignore_none(sub_vis_list, facet_list))
                vis_results.append(gather_image_iteration_results(facet_vis_results,
                                                                  template_model_imagelist[freqwin]))
            results_vislist.append(sum_invert_results(vis_results))

    return results_vislist


def residual_list_serial_workflow(vis, model_imagelist, context='2d', **kwargs):
    """ Create a graph to calculate residual image using w stacking and faceting

    :param context:
    :param vis:
    :param model_imagelist: Model used to determine image parameters
    :param vis:
    :param model_imagelist: Model used to determine image parameters
    :param kwargs: Parameters for functions in components
    :return:
    """
    model_vis = zero_list_serial_workflow(vis)
    model_vis = predict_list_serial_workflow(model_vis, model_imagelist, context=context, **kwargs)
    residual_vis = subtract_list_serial_workflow(vis, model_vis)
    return invert_list_serial_workflow(residual_vis, model_imagelist, dopsf=False, normalize=True, context=context,
                                       **kwargs)


def restore_list_serial_workflow(model_imagelist, psf_imagelist, residual_imagelist, **kwargs):
    """ Create a graph to calculate the restored image

    :param model_imagelist: Model list
    :param psf_imagelist: PSF list
    :param residual_imagelist: Residual list
    :param kwargs: Parameters for functions in components
    :return:
    """
    return [restore_cube(model_imagelist[i], psf_imagelist[i][0],
                         residual_imagelist[i][0], **kwargs)
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
    
    def deconvolve(dirty, psf, model, facet, gthreshold):
        import time
        starttime = time.time()
        if prefix == '':
            lprefix = "facet %d" % facet
        else:
            lprefix = "%s, facet %d" % (prefix, facet)
        
        nmoments = get_parameter(kwargs, "nmoments", 0)
        
        if nmoments > 0:
            moment0 = calculate_image_frequency_moments(dirty)
            this_peak = numpy.max(numpy.abs(moment0.data[0, ...])) / dirty.data.shape[0]
        else:
            this_peak = numpy.max(numpy.abs(dirty.data[0, ...]))
        
        if this_peak > 1.1 * gthreshold:
            log.info(
                "deconvolve_list_serial_workflow %s: cleaning - peak %.6f > 1.1 * threshold %.6f" % (lprefix, this_peak,
                                                                                                gthreshold))
            kwargs['threshold'] = gthreshold
            result, _ = deconvolve_cube(dirty, psf, prefix=lprefix, **kwargs)
            
            if result.data.shape[0] == model.data.shape[0]:
                result.data += model.data
            else:
                log.warning(
                    "deconvolve_list_serial_workflow %s: Initial model %s and clean result %s do not have the same shape" %
                    (lprefix, str(model.data.shape[0]), str(result.data.shape[0])))
            
            flux = numpy.sum(result.data[0, 0, ...])
            log.info('### %s, %.6f, %.6f, True, %.3f # cycle, facet, peak, cleaned flux, clean, time?'
                     % (lprefix, this_peak, flux, time.time() - starttime))
            
            return result
        else:
            log.info("deconvolve_list_serial_workflow %s: Not cleaning - peak %.6f <= 1.1 * threshold %.6f" % (
                lprefix, this_peak,
                gthreshold))
            log.info('### %s, %.6f, %.6f, False, %.3f # cycle, facet, peak, cleaned flux, clean, time?'
                     % (lprefix, this_peak, 0.0, time.time() - starttime))
            
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
    nmoments = get_parameter(kwargs, "nmoments", 0)
    use_moment0 = nmoments > 0
    
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


def weight_list_serial_workflow(vis_list, model_imagelist, weighting='uniform', **kwargs):
    """ Weight the visibility data

    :param vis_list:
    :param model_imagelist: Model required to determine weighting parameters
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
    
    return [weight_vis(vis_list[i], model_imagelist[i])
            for i in range(len(vis_list))]


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
    
    return [subtract_vis(vis=vis_list[i],
                         model_vis=model_vislist[i])
            for i in range(len(vis_list))]
