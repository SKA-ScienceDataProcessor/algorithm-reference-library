""" Common functions converted to Dask.bags graphs. `Dask <http://dask.pydata.org/>`_ is a python-based flexible
parallel computing library for analytic computing.

Note that all parameters here should be passed using the kwargs mechanism.
"""
import numpy
import logging
import collections

from dask import bag

from arl.data.data_models import Image
from arl.graphs.graphs import sum_invert_results
from arl.imaging import *
from arl.visibility.base import copy_visibility
from arl.image.operations import create_image_from_array
from arl.imaging.imaging_context import imaging_context
from arl.visibility.operations import concat_visibility
from arl.image.deconvolution import deconvolve_cube, restore_cube

log = logging.getLogger(__name__)

def safe_predict_list(vis_list, model, predict=predict_2d, **kwargs):
    """ Predicts a list of visibilities to obtain a list of visibilities
    
    :param vis_list:
    :param model:
    :param predict:
    :param kwargs:
    :return: List of visibilities
    """
    assert isinstance(vis_list, collections.Iterable), vis_list
    assert isinstance(model, Image), "Model is not an image: %s" % model
    result = list()
    for v in vis_list:
        if v is not None:
            predicted = copy_visibility(v)
            result.append(predict(predicted, model, **kwargs))
    return result


def safe_invert_list(vis_list, model, invert=invert_2d, *args, **kwargs):
    """Invert a list of visibilities to obtain a list of (Image, weight) tuples
    
    :param vis_list:
    :param model:
    :param invert:
    :param args:
    :param kwargs:
    :return: List of (Image, weight) tuples
    """
    result = list()
    assert isinstance(vis_list, collections.Iterable), vis_list
    assert isinstance(model, Image), "Model is not an image: %s" % model
    for v in vis_list:
        if v is not None:
            result.append(invert(v, model, *args, **kwargs))
    return result


def sum_invert_results(invert_list, normalize=True):
    """Sum a list of invert results, optionally normalizing at the end

    :param invert_list: List of results from invert: Image, weight tuples
    :param normalize: Normalize by the sum of weights
    """
    assert isinstance(invert_list, collections.Iterable), invert_list
    result = None
    weight = None
    for i, a in enumerate(invert_list):
        assert isinstance(a[0], Image), "Item is not an image: %s" % str(a[0])
        if i == 0:
            result = create_image_from_array(a[0].data * a[1], a[0].wcs, a[0].polarisation_frame)
            weight = a[1]
        else:
            result.data += a[0].data * a[1]
            weight += a[1]

    assert weight is not None and result is not None, "No valid images found"
    
    if normalize:
        result = normalize_sumwt(result, weight)
    
    return result, weight


def invert_bag(vis_bag, model, dopsf=False, context='2d', **kwargs):
    """ Inverts a bag of visibilities to create a bag of (image, weight) tuples
    
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
        map(sum_invert_results)


def predict_bag(vis_bag, model, context='2d', **kwargs):
    """Predicts a bag of visibilities to obtain a bag of visibilities.

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

def deconvolve_bag(dirty_bag, psf_bag, **kwargs):
    """ Deconvolve a bag of images to obtain a bag of models
    
    :param dirty_bag:
    :param psf_bag:
    :param kwargs:
    :return: Bag of Images
    """

    def deconvolve(dirty_psf, **kwargs):
        result = deconvolve_cube(dirty_psf[0][0], dirty_psf[1][0], **kwargs)
        return result[0]
    
    return bag.zip(dirty_bag, psf_bag).map(deconvolve, **kwargs)


def restore_bag(comp_bag, psf_bag, residual_bag, **kwargs):
    """ Deconvolve a bag of images to obtain a bag of models

    :param dirty_bag:
    :param psf_bag:
    :param kwargs:
    :return: Bag of Images
    """
    
    def restore(comp_psf_residual, **kwargs):
        return restore_cube(comp_psf_residual[0], comp_psf_residual[1][0], comp_psf_residual[2][0], **kwargs)
    
    return bag.zip(comp_bag, psf_bag, residual_bag).map(restore, **kwargs)

def residual_visibility_bag(vis_bag, model_vis_bag, **kwargs):
    def subtract_vis(vis, model_vis):
        residual_vis = copy_visibility(vis)
        residual_vis.data['vis']-=model_vis.data['vis']
        return residual_vis
        
    return vis_bag.map(subtract_vis, model_vis_bag)