""" Common functions converted to Dask.bags graphs. `Dask <http://dask.pydata.org/>`_ is a python-based flexible
parallel computing library for analytic computing.
Note that all parameters here should be passed using the kwargs mechanism. The exceptions
are those needed to define the size of a graph. Since delayed graphs are not Iterable
by default, it is necessary to use the nout= parameter to delayed to specify the
graph size.

Construction of the graphs requires that the number of nodes (e.g. w slices or time-slices) be known at construction,
rather than execution. To counteract this, at run time, a given node should be able to act as a no-op. This is a
workaround only.

"""
import numpy
import logging
import collections

from arl.graphs.graphs import sum_invert_results
from arl.imaging import *
from arl.visibility.base import copy_visibility
from arl.image.operations import create_image_from_array
from arl.imaging.imaging_context import imaging_context
from arl.visibility.operations import concat_visibility

log = logging.getLogger(__name__)

def safe_predict_list(vis_list, model, predict=predict_2d, **kwargs):
    assert isinstance(vis_list, collections.Iterable), vis_list
    result = list()
    for v in vis_list:
        if v is not None:
            predicted = copy_visibility(v)
            result.append(predict(predicted, model, **kwargs))
    return result


def safe_invert_list(vis_list, model, invert=invert_2d, *args, **kwargs):
    result = list()
    assert isinstance(vis_list, collections.Iterable), vis_list
    for v in vis_list:
        if v is not None:
            result.append(invert(v, model, *args, **kwargs))
    return result


def sum_invert_results(invert_list, normalize=True):
    """Sum a set of invert results, optionally normalizing at the end

    :param invert_list: List of results from invert: Image, weight tuples
    :param normalize: Normalize by the sum of weights
    """
    assert isinstance(invert_list, collections.Iterable), invert_list
    for i, a in enumerate(invert_list):
        if i == 0:
            result = create_image_from_array(a[0].data * a[1], a[0].wcs, a[0].polarisation_frame)
            weight = a[1]
        else:
            result.data += a[0].data * a[1]
            weight += a[1]
    
    if normalize:
        result = normalize_sumwt(result, weight)
    
    return result, weight


def invert_bag(vis_bag, model, dopsf=False, context='2d', **kwargs):
    """
    
    :param vis_bag:
    :param model:
    :param context:
    :param kwargs:
    :return:
    """
    c = imaging_context(context)
    log.info('Imaging context is %s' % c)
    assert c['scatter'] is not None
    return vis_bag.\
        map(c['scatter'], **kwargs). \
        map(safe_invert_list, model, c['invert'], dopsf=dopsf, **kwargs). \
        flatten().\
        reduction(sum_invert_results, sum_invert_results)


def predict_bag(vis_bag, model, context='2d', **kwargs):
    """

    :param vis_bag:
    :param model:
    :param context:
    :param kwargs:
    :return:
    """
    c = imaging_context(context)
    assert c['scatter'] is not None and c['gather'] is not None
    return vis_bag.\
        map(c['scatter'], **kwargs). \
        map(safe_predict_list, model, c['predict'], **kwargs).\
        map(concat_visibility)
