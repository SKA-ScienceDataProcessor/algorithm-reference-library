""" Imaging context definitions, potentially shared by other workflows

"""

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