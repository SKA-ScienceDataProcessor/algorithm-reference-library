""" Imaging context definitions, potentially shared by other workflows

"""
import numpy

import logging

from processing_components.imaging.base import normalize_sumwt
from processing_components.visibility.base import copy_visibility
from processing_components.image.operations import copy_image, calculate_image_frequency_moments

log = logging.getLogger(__name__)

from processing_components.imaging.base import predict_2d, invert_2d
from processing_components.visibility.iterators import vis_null_iter, vis_timeslice_iter, vis_wslice_iter
from processing_components.imaging.timeslice_single import predict_timeslice_single, invert_timeslice_single
from processing_components.imaging.wstack_single import predict_wstack_single, invert_wstack_single


def imaging_contexts():
    """Contains all the context information for imaging
    
    The fields are:
        predict: Predict function to be used
        invert: Invert function to be used
        image_iterator: Iterator for traversing images
        vis_iterator: Iterator for traversing visibilities
        inner: The innermost axis
    
    :return:
    """
    contexts = {'2d': {'predict': predict_2d,
                       'invert': invert_2d,
                       'vis_iterator': vis_null_iter},
                'wprojection': {'predict': predict_2d,
                       'invert': invert_2d,
                       'vis_iterator': vis_null_iter},
                'facets': {'predict': predict_2d,
                           'invert': invert_2d,
                           'vis_iterator': vis_null_iter},
                'facets_timeslice': {'predict': predict_timeslice_single,
                                     'invert': invert_timeslice_single,
                                     'vis_iterator': vis_timeslice_iter},
                'facets_wstack': {'predict': predict_wstack_single,
                                  'invert': invert_wstack_single,
                                  'vis_iterator': vis_wslice_iter},
                'timeslice': {'predict': predict_timeslice_single,
                              'invert': invert_timeslice_single,
                              'vis_iterator': vis_timeslice_iter},
                'wstack': {'predict': predict_wstack_single,
                           'invert': invert_wstack_single,
                           'vis_iterator': vis_wslice_iter}}
    
    return contexts


def imaging_context(context='2d'):
    contexts = imaging_contexts()
    assert context in contexts.keys(), context
    return contexts[context]


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


def threshold_list(imagelist, threshold, fractional_threshold, use_moment0=True, prefix=''):
    """ Find actual threshold for list of results, optionally using moment 0

    :param imagelist:
    :param threshold: Absolute threshold
    :param fractional_threshold: Fractional  threshold
    :param use_moment0: Use moment 0 for threshold
    :return:
    """
    peak = 0.0
    for result in imagelist:
        if use_moment0:
            moments = calculate_image_frequency_moments(result)
            peak = max(peak, numpy.max(numpy.abs(moments.data[0, ...] / result.shape[0])))
        else:
            peak = max(peak, numpy.max(numpy.abs(result.data)))
    
    actual = max(peak * fractional_threshold, threshold)
    
    if use_moment0:
        log.info("threshold_list %s: peak in moment 0 = %.6f, threshold will be %.6f" % (prefix, peak, actual))
    else:
        log.info("threshold_list %s: peak = %.6f, threshold will be %.6f" % (prefix, peak, actual))
    
    return actual