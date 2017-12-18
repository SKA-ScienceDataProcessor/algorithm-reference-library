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
                'timeslice': {'predict': predict_timeslice, 'invert': invert_timeslice,
                              'scatter': None, 'gather': None},
                'timeslice_single': {'predict': predict_timeslice_single, 'invert': invert_timeslice_single,
                                     'scatter': visibility_scatter_time, 'gather': visibility_gather_time},
                'wstack': {'predict': predict_wstack, 'invert': invert_wstack,
                           'scatter': None, 'gather': None},
                'wstack_single': {'predict': predict_wstack_single, 'invert': invert_wstack_single,
                                  'scatter': visibility_scatter_w, 'gather': visibility_gather_w}}
    
    return contexts


def imaging_context(context='2d'):
    contexts = imaging_contexts()
    assert context in contexts.keys(), context
    return contexts[context]


def invert_context(vis, model, dopsf=False, normalize=True, context='2d', **kwargs):
    """ Invert selected by context
    
    :param vis:
    :param model:
    :param dopsf:
    :param context:
    :param kwargs:
    :return:
    """
    log.debug('invert_context: Imaging context is %s, using function %s'
              % (context, imaging_context(context)['invert']))
    return imaging_context(context)['invert'](vis, model, dopsf=dopsf, normalize=normalize, **kwargs)


def predict_context(vis, model, context='2d', **kwargs):
    """ Predict selected by context

    :param vis:
    :param model:
    :param context:
    :param kwargs:
    :return:
    """
    log.debug('predict_context: Imaging context is %s, using function %s'
              % (context, imaging_context(context)['predict']))
    return imaging_context(context)['predict'](vis, model, **kwargs)
