""" Common functions converted to Dask.bags graphs. `Dask <http://dask.pydata.org/>`_ is a python-based flexible
parallel computing library for analytic computing.

Note that all parameters here should be passed using the kwargs mechanism.
"""
import collections
import logging

from dask import bag

from arl.calibration.operations import qa_gaintable
from arl.data.data_models import Image
from arl.image.deconvolution import deconvolve_cube, restore_cube
from arl.image.operations import create_image_from_array, qa_image
from arl.imaging import predict_2d, invert_2d, normalize_sumwt
from arl.imaging.imaging_context import imaging_context
from arl.visibility.base import copy_visibility
from arl.visibility.operations import concatenate_visibility, subtract_visibility, qa_visibility

log = logging.getLogger(__name__)


def safe_predict_list(vis_list, model, predict=predict_2d, **kwargs):
    """ Predicts a list of visibilities to obtain a list of visibilities
    
    Can be used in bag.map()
    
    :param vis_list:
    :param model:
    :param predict:
    :param kwargs:
    :return: List of visibilities
    """
    assert isinstance(vis_list, collections.Iterable), "Visibility list is not Iterable: %s" % str(vis_list)
    
    assert isinstance(model, Image), "Model is not an image: %s" % model
    
    result = list()
    for v in vis_list:
        if v is not None:
            predicted = copy_visibility(v)
            result.append(predict(predicted, model, **kwargs))
    
    assert len(result) > 0, "Visibility after concatenation is empty, input list is %s" % str(vis_list)
    
    return result


def safe_invert_list(vis_list, model, invert=invert_2d, *args, **kwargs):
    """Invert a list of visibilities to obtain a list of (Image, weight) tuples
    
    Can be used in bag.map()
    
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


def sum_invert_bag_results(invert_list, normalize=True):
    """Sum a list of invert results, optionally normalizing at the end

    Can be used in bag.map()
    
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


def invert_bag(vis_bag, model_bag, dopsf=False, context='2d', **kwargs):
    """ Construct a bag to invert a bag of visibilities to a bag of (image, weight) tuples
    
    Call directly - don't use via bag.map
    
    :param vis_bag:
    :param model:
    :param context:
    :param kwargs:
    :return:
    """
    c = imaging_context(context)
    log.info('Imaging context is %s' % c)
    assert c['scatter'] is not None
    return vis_bag. \
        map(c['scatter'], **kwargs). \
        map(safe_invert_list, model_bag, c['invert'], dopsf=dopsf, **kwargs). \
        map(sum_invert_bag_results)


def predict_bag(vis_bag, model_bag, context='2d', **kwargs):
    """Construct a bag to predict a bag of visibilities.
    
    The vis_bag is scatter appropriately, the predict is applied, and the data then
    concatenated. The sort order of the data is not necessarily preserved.

    Call directly - don't use via bag.map
    
    :param vis_bag:
    :param model:
    :param context:
    :param kwargs:
    :return:
    """
    c = imaging_context(context)
    assert c['scatter'] is not None
    
    return vis_bag. \
        map(copy_visibility, zero=True). \
        map(c['scatter'], **kwargs). \
        map(safe_predict_list, model_bag, c['predict'], **kwargs). \
        map(concatenate_visibility)


def deconvolve_bag(dirty_bag, psf_bag, **kwargs):
    """ Deconvolve a bag of images to obtain a bag of models
    
    Call directly - don't use via bag.map
    
    :param dirty_bag:
    :param psf_bag:
    :param kwargs:
    :return: Bag of Images
    """
    
    def deconvolve(dp_zip, **kwargs):
        # The dirty and psf are actually (Image, weight) tuples.
        result = deconvolve_cube(dp_zip[0][0], dp_zip[1][0], **kwargs)
        return result[0]
    
    # We zip up the dirty and psf bags and call the deconvolve adapter
    return bag.zip(dirty_bag, psf_bag).map(deconvolve, **kwargs)


def restore_bag(comp_bag, psf_bag, residual_bag, **kwargs):
    """ Restore a bag of images to obtain a bag of restored images

    Call directly - don't use via bag.map
    
    :param dirty_bag:
    :param psf_bag:
    :param kwargs:
    :return: Bag of Images
    """
    
    def restore(cpr_zip, **kwargs):
        # The comp is just an Image, while the dirty and psf are actually (Image, weight) tuples.
        return restore_cube(cpr_zip[0], cpr_zip[1][0], cpr_zip[2][0], **kwargs)
    
    return bag.zip(comp_bag, psf_bag, residual_bag).map(restore, **kwargs)


def residual_image_bag(vis_bag, model_image_bag, context='2d', **kwargs):
    """Calculate residual images

    Call directly - don't use via bag.map
    
    :param vis_bag: Bag containing visibilities
    :param model_image_bag: Model images, one per visibility in vis_bag
    :param kwargs:
    :return:
    """
    model_vis_bag = predict_bag(vis_bag, model_image_bag, context=context, **kwargs)
    res_vis_bag = residual_vis_bag(vis_bag, model_vis_bag)
    return invert_bag(res_vis_bag, model_image_bag, context=context, **kwargs)


def residual_vis_bag(vis_bag, model_vis_bag):
    """Calculate residual visibility

    Call directly - don't use via bag.map
    
    :param vis_bag: Bag containing visibilities
    :param model_image_bag: Model images, one per visibility in vis_bag
    :param kwargs:
    :return:
    """
    
    def subtract_vis_zip(vis_zip_bag):
        return subtract_visibility(vis_zip_bag[0], vis_zip_bag[1])
    
    return bag.zip(vis_bag, model_vis_bag).map(subtract_vis_zip)


def qa_visibility_bag(vis, context=''):
    """ Print qa on the visibilities, use this in a sequence of bag operations
    
    Can be used in bag.map() as a passthru
    
    :param vis:
    :return:
    """
    s = qa_visibility(vis, context=context)
    log.info(s)
    print(s)
    return vis


def qa_image_bag(im, context=''):
    """ Print qa on images, use this in a sequence of bag operations

    Can be used in bag.map() as a passthru
    
    :param im:
    :return:
    """
    s = qa_image(im, context=context)
    log.info(s)
    print(s)
    return im


def qa_gaintable_bag(gt, context=''):
    """ Print qa on gaintables, use this in a sequence of bag operations

    Can be used in bag.map() as a passthru
    
    :param gt:
    :return:
    """
    s = qa_gaintable(gt, context=context)
    log.info(s)
    print(s)
    return gt
