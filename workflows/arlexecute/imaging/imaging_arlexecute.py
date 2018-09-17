""" Imaging workflows using arlexecute and processing_components functions.

Construction of the components requires that the number of nodes (e.g. w slices or time-slices) be known at construction,
rather than execution. To counteract this, at run time, a given node should be able to act as a no-op. We use None
to denote a null node.

The actual imaging code executed eventually is specified by the context variable (see processing_library.imaging.imaging)context.
These are the same as executed in the imaging framework.

"""

import collections
import logging

import numpy

from data_models.memory_data_models import Image
from data_models.parameters import get_parameter
from processing_library.image.operations import copy_image, create_empty_image_like
from workflows.shared.imaging.imaging_shared import imaging_context
from workflows.shared.imaging.imaging_shared import sum_invert_results, remove_sumwt, sum_predict_results, \
    threshold_list
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.image.deconvolution import deconvolve_cube, restore_cube
from wrappers.arlexecute.image.gather_scatter import image_scatter_facets, image_gather_facets, \
    image_scatter_channels, image_gather_channels
from wrappers.arlexecute.image.operations import calculate_image_frequency_moments
from wrappers.arlexecute.imaging.weighting import weight_visibility
from wrappers.arlexecute.visibility.base import copy_visibility
from wrappers.arlexecute.visibility.gather_scatter import visibility_scatter, visibility_gather

log = logging.getLogger(__name__)


def predict_list_arlexecute_workflow(vis_list, model_imagelist, vis_slices=1, facets=1, context='2d', **kwargs):
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
    
    if facets % 2 == 0 or facets == 1:
        actual_number_facets = facets
    else:
        actual_number_facets = facets - 1
    
    def predict_ignore_none(vis, model):
        if vis is not None:
            return predict(vis, model, context=context, facets=facets, vis_slices=vis_slices, **kwargs)
        else:
            return None
    
    image_results_list_list = list()
    # Loop over all frequency windows
    for freqwin, vis_list in enumerate(vis_list):
        # Create the graph to divide an image into facets. This is by reference.
        facet_lists = arlexecute.execute(image_scatter_facets, nout=actual_number_facets ** 2)(model_imagelist[freqwin],
                                                                                               facets=facets)
        # Create the graph to divide the visibility into slices. This is by copy.
        sub_vis_lists = arlexecute.execute(visibility_scatter, nout=vis_slices)(vis_list, vis_iter, vis_slices)
        
        facet_vis_lists = list()
        # Loop over sub visibility
        for sub_vis_list in sub_vis_lists:
            facet_vis_results = list()
            # Loop over facets
            for facet_list in facet_lists:
                # Predict visibility for this subvisibility from this facet
                facet_vis_list = arlexecute.execute(predict_ignore_none, pure=True, nout=1)(sub_vis_list,
                                                                                            facet_list)
                facet_vis_results.append(facet_vis_list)
            # Sum the current sub-visibility over all facets
            facet_vis_lists.append(arlexecute.execute(sum_predict_results)(facet_vis_results))
        # Sum all sub-visibilties
        image_results_list_list.append(
            arlexecute.execute(visibility_gather, nout=1)(facet_vis_lists, vis_list, vis_iter))
    
    return image_results_list_list


def invert_list_arlexecute_workflow(vis_list, template_model_imagelist, dopsf=False, normalize=True,
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
    
    # Loop over all vis_lists independently
    results_vislist = list()
    for freqwin, vis_list in enumerate(vis_list):
        # Create the graph to divide an image into facets. This is by reference.
        facet_lists = arlexecute.execute(image_scatter_facets, nout=actual_number_facets ** 2)(template_model_imagelist[
                                                                                                   freqwin],
                                                                                               facets=facets)
        # Create the graph to divide the visibility into slices. This is by copy.
        sub_vis_lists = arlexecute.execute(visibility_scatter, nout=vis_slices)(vis_list, vis_iter,
                                                                                vis_slices=vis_slices)
        
        # Iterate within each vis_list
        vis_results = list()
        for sub_vis_list in sub_vis_lists:
            facet_vis_results = list()
            for facet_list in facet_lists:
                facet_vis_results.append(
                    arlexecute.execute(invert_ignore_none, pure=True)(sub_vis_list, facet_list))
            vis_results.append(arlexecute.execute(gather_image_iteration_results, nout=1)(facet_vis_results,
                                                                                          template_model_imagelist[
                                                                                              freqwin]))
        results_vislist.append(arlexecute.execute(sum_invert_results)(vis_results))
    
    return results_vislist


def residual_list_arlexecute_workflow(vis, model_imagelist, context='2d', **kwargs):
    """ Create a graph to calculate residual image using w stacking and faceting

    :param context: 
    :param vis:
    :param model_imagelist: Model used to determine image parameters
    :param vis:
    :param model_imagelist: Model used to determine image parameters
    :param kwargs: Parameters for functions in components
    :return:
    """
    model_vis = zero_list_arlexecute_workflow(vis)
    model_vis = predict_list_arlexecute_workflow(model_vis, model_imagelist, context=context, **kwargs)
    residual_vis = subtract_list_arlexecute_workflow(vis, model_vis)
    return invert_list_arlexecute_workflow(residual_vis, model_imagelist, dopsf=False, normalize=True, context=context,
                                           **kwargs)


def restore_list_arlexecute_workflow(model_imagelist, psf_imagelist, residual_imagelist, **kwargs):
    """ Create a graph to calculate the restored image

    :param model_imagelist: Model list
    :param psf_imagelist: PSF list
    :param residual_imagelist: Residual list
    :param kwargs: Parameters for functions in components
    :return:
    """
    psf_list = arlexecute.execute(remove_sumwt)(psf_imagelist)
    residual_list = arlexecute.execute(remove_sumwt)(residual_imagelist)
    print('model_imagelist', model_imagelist)
    print('psf_imagelist', psf_imagelist)
    print('residual_imagelist', residual_imagelist)
    return [arlexecute.execute(restore_cube)(model_imagelist[i], psf_list[i],
                                             residual_list[i], **kwargs)
            for i, _ in enumerate(residual_imagelist)]

def deconvolve_list_arlexecute_workflow(dirty_list, psf_list, model_imagelist, prefix='', **kwargs):
    """Create a graph for deconvolution, adding to the model

    :param dirty_list:
    :param psf_list:
    :param model_imagelist:
    :param kwargs: Parameters for functions in components
    :return: (graph for the deconvolution, graph for the flat)
    """
    nchan = len(dirty_list)
    nmoments = get_parameter(kwargs, "nmoments", 0)

    def deconvolve(dirty, psf, model, facet, gthreshold):
        if prefix == '':
            lprefix = "facet %d" % facet
        else:
            lprefix = "%s, facet %d" % (prefix, facet)
        
        if nmoments > 0:
            moment0 = calculate_image_frequency_moments(dirty)
            this_peak = numpy.max(numpy.abs(moment0.data[0, ...])) / dirty.data.shape[0]
        else:
            this_peak = numpy.max(numpy.abs(dirty.data[0, ...]))
        
        if this_peak > 1.1 * gthreshold:
            # log.info(
            #     "deconvolve_list_arlexecute_workflow %s: cleaning - peak %.6f > 1.1 * threshold %.6f" % (
            #     lprefix, this_peak,
            #     gthreshold))
            kwargs['threshold'] = gthreshold
            result, _ = deconvolve_cube(dirty, psf, prefix=lprefix, **kwargs)
            
            if result.data.shape[0] == model.data.shape[0]:
                result.data += model.data
            # else:
            #     log.warning(
            #         "deconvolve_list_arlexecute_workflow %s: Initial model %s and clean result %s do not have the same shape" %
            #         (lprefix, str(model.data.shape[0]), str(result.data.shape[0])))
            #
            flux = numpy.sum(result.data[0, 0, ...])
            # log.info('### %s, %.6f, %.6f, True, # cycle, facet, peak, cleaned flux, clean'
            #          % (lprefix, this_peak, flux[0]))
            #
            return result
        else:
            # log.info("deconvolve_list_arlexecute_workflow %s: Not cleaning - peak %.6f <= 1.1 * threshold %.6f" % (
            #     lprefix, this_peak,
            #     gthreshold))
            # log.info('### %s, %.6f, %.6f, False, %.3f # cycle, facet, peak, cleaned flux, clean'
            #          % (lprefix, this_peak, 0.0))
            
            return copy_image(model)
    
    deconvolve_facets = get_parameter(kwargs, 'deconvolve_facets', 1)
    deconvolve_overlap = get_parameter(kwargs, 'deconvolve_overlap', 0)
    deconvolve_taper = get_parameter(kwargs, 'deconvolve_taper', None)
    if deconvolve_overlap > 0:
        deconvolve_number_facets = (deconvolve_facets - 2) ** 2
    else:
        deconvolve_number_facets = deconvolve_facets ** 2
    
    model_imagelist = arlexecute.execute(image_gather_channels, nout=1)(model_imagelist)
    
    # Scatter the separate channel images into deconvolve facets and then gather channels for each facet.
    # This avoids constructing the entire spectral cube.
    dirty_list_trimmed = arlexecute.execute(remove_sumwt, nout=nchan)(dirty_list)
    scattered_channels_facets_dirty_list = \
        [arlexecute.execute(image_scatter_facets, nout=deconvolve_number_facets)(d, facets=deconvolve_facets,
                                                                                 overlap=deconvolve_overlap,
                                                                                 taper=deconvolve_taper)
         for d in dirty_list_trimmed]
    
    # Now we do a transpose and gather
    scattered_facets_list = [
        arlexecute.execute(image_gather_channels, nout=1)([scattered_channels_facets_dirty_list[chan][facet]
                                                           for chan in range(nchan)])
        for facet in range(deconvolve_number_facets)]
    
    psf_list_trimmed = arlexecute.execute(remove_sumwt, nout=nchan)(psf_list)
    psf_list_trimmed = arlexecute.execute(image_gather_channels, nout=1)(psf_list_trimmed)
    
    scattered_model_imagelist = \
        arlexecute.execute(image_scatter_facets, nout=deconvolve_number_facets)(model_imagelist,
                                                                                facets=deconvolve_facets,
                                                                                overlap=deconvolve_overlap)
    
    # Work out the threshold. Need to find global peak over all dirty_list images
    threshold = get_parameter(kwargs, "threshold", 0.0)
    fractional_threshold = get_parameter(kwargs, "fractional_threshold", 0.1)
    nmoments = get_parameter(kwargs, "nmoments", 0)
    use_moment0 = nmoments > 0
    
    # Find the global threshold. This uses the peak in the average on the frequency axis since we
    # want to use it in a stopping criterion in a moment clean
    global_threshold = arlexecute.execute(threshold_list, nout=1)(scattered_facets_list, threshold,
                                                                  fractional_threshold,
                                                                  use_moment0=use_moment0, prefix=prefix)
    
    facet_list = numpy.arange(deconvolve_number_facets).astype('int')
    scattered_results_list = [
        arlexecute.execute(deconvolve, nout=1)(d, psf_list_trimmed, m, facet, global_threshold)
        for d, m, facet in zip(scattered_facets_list, scattered_model_imagelist, facet_list)]
    
    # Gather the results back into one image, correcting for overlaps as necessary. The taper function is is used to
    # feather the facets together
    gathered_results_list = arlexecute.execute(image_gather_facets, nout=1)(scattered_results_list, model_imagelist,
                                                                            facets=deconvolve_facets,
                                                                            overlap=deconvolve_overlap,
                                                                            taper=deconvolve_taper)
    flat_list = arlexecute.execute(image_gather_facets, nout=1)(scattered_results_list, model_imagelist,
                                                                facets=deconvolve_facets, overlap=deconvolve_overlap,
                                                                taper=deconvolve_taper, return_flat=True)
    
    return arlexecute.execute(image_scatter_channels, nout=nchan)(gathered_results_list, subimages=nchan), flat_list


def deconvolve_list_channel_arlexecute_workflow(dirty_list, psf_list, model_imagelist, subimages, **kwargs):
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
    
    output = arlexecute.execute(create_empty_image_like, nout=1, pure=True)(model_imagelist)
    dirty_lists = arlexecute.execute(image_scatter_channels, nout=subimages, pure=True)(dirty_list[0],
                                                                                        subimages=subimages)
    results = [arlexecute.execute(deconvolve_subimage)(dirty_list, psf_list[0])
               for dirty_list in dirty_lists]
    result = arlexecute.execute(image_gather_channels, nout=1, pure=True)(results, output, subimages=subimages)
    return arlexecute.execute(add_model, nout=1, pure=True)(result, model_imagelist)


def weight_list_arlexecute_workflow(vis_list, model_imagelist, weighting='uniform', **kwargs):
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
    
    return [arlexecute.execute(weight_vis, pure=True, nout=1)(vis_list[i], model_imagelist[i])
            for i in range(len(vis_list))]


def zero_list_arlexecute_workflow(vis_list):
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
    
    return [arlexecute.execute(zero, pure=True, nout=1)(v) for v in vis_list]


def subtract_list_arlexecute_workflow(vis_list, model_vislist):
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
    
    return [arlexecute.execute(subtract_vis, pure=True, nout=1)(vis=vis_list[i],
                                                                model_vis=model_vislist[i])
            for i in range(len(vis_list))]
