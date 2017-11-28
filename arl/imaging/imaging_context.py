"""Manages the imaging context. This take a strings and returns a dictionary containing:
 * Predict function
 * Invert function
 * Scatter function
 * Gather function

"""

import logging

from arl.imaging import *
from arl.visibility.gather_scatter import visibility_gather_index, visibility_scatter_index, \
    visibility_gather_time, visibility_scatter_time, visibility_gather_w, visibility_scatter_w

log = logging.getLogger(__name__)


def imaging_contexts():
    """Contains all the context information for imaging
    
    :return:
    """
    contexts = {'2d': {'predict': predict_2d, 'invert': invert_2d,
                       'scatter': visibility_scatter_index, 'gather': visibility_gather_index},
                'slice': {'predict': predict_2d, 'invert': invert_2d,
                          'scatter': visibility_scatter_index, 'gather': visibility_gather_index},
                'timeslice': {'predict': predict_timeslice_single, 'invert': invert_timeslice_single,
                              'scatter': visibility_scatter_time, 'gather': visibility_gather_time},
                'wstack': {'predict': predict_wstack_single, 'invert': invert_wstack_single,
                           'scatter': visibility_scatter_w, 'gather': visibility_gather_w}}
    return contexts


def imaging_context(context='2d'):
    contexts = imaging_contexts()
    assert context in contexts.keys()
    return contexts[context]
