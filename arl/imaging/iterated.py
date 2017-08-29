
#
"""
Functions that distributes predict and invert using either just loops or parallel execution
"""
import numpy

from arl.data.data_models import Visibility, Image
from arl.imaging import normalize_sumwt
from arl.image.iterators import raster_iter
from arl.image.operations import create_empty_image_like
from arl.visibility.iterators import vis_slice_iter
from arl.visibility.base import copy_visibility, create_visibility_from_rows
from arl.visibility.coalesce import coalesce_visibility
from arl.imaging.base import predict_2d_base, predict_2d, invert_2d_base, invert_2d

import logging

log = logging.getLogger(__name__)


def invert_with_vis_iterator(vis: Visibility, im: Image, dopsf=False, normalize=True, vis_iter=vis_slice_iter,
                             invert=invert_2d, **kwargs):
    """ Invert using a specified iterator and invert
    
    This knows about the structure of invert in different execution frameworks but not
    anything about the actual processing.

    :param vis:
    :param im:
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    :param kwargs:
    :return:
    """
    resultimage = create_empty_image_like(im)
    
    if type(vis) is not Visibility:
        svis = coalesce_visibility(vis, **kwargs)
    else:
        svis = vis

    
    i = 0
    for rows in vis_iter(svis, **kwargs):
        if numpy.sum(rows) and svis is not None:
            visslice = create_visibility_from_rows(svis, rows)
            workimage, sumwt = invert(visslice, im, dopsf, normalize=False, **kwargs)
            resultimage.data += workimage.data
            if i == 0:
                totalwt = sumwt
            else:
                totalwt += sumwt
            i += 1
        
    if normalize:
        resultimage = normalize_sumwt(resultimage, totalwt)
    
    return resultimage, totalwt


def predict_with_vis_iterator(vis: Visibility, model: Image, vis_iter=vis_slice_iter,
                              predict=predict_2d, **kwargs) -> Visibility:
    """Iterate through prediction in chunks
    
    This knows about the structure of predict in different execution frameworks but not
    anything about the actual processing.
    
    """
    log.debug("predict_with_vis_iterator: Processing chunks")
    if type(vis) is not Visibility:
        svis = coalesce_visibility(vis, **kwargs)
    else:
        svis = vis
        
    # Do each chunk in turn
    for rows in vis_iter(svis, **kwargs):
        if numpy.sum(rows) and svis is not None:
            visslice = create_visibility_from_rows(svis, rows)
            visslice.data['vis'][...] = 0.0
            visslice = predict(visslice, model, **kwargs)
            svis.data['vis'][rows] += visslice.data['vis']
    return svis


def predict_with_image_iterator(vis: Visibility, model: Image, image_iterator=raster_iter,
                                predict_function=predict_2d_base, **kwargs) -> Visibility:
    """ Predict using image partitions, calling specified predict function

    :param vis: Visibility to be predicted
    :param model: model image
    :param image_iterator: Image iterator used to access the image
    :param predict_function: Function to be used for prediction (allows nesting)
    :return: resulting visibility (in place works)
    """
    log.info("predict_with_image_iterator: Predicting by image partitions")
    result = copy_visibility(vis)
    for dpatch in image_iterator(model, **kwargs):
        result.data['vis'][...] = 0.0
        result = predict_function(result, dpatch, **kwargs)
        vis.data['vis'] += result.data['vis']
    return vis


def invert_with_image_iterator(vis, im, image_iterator=raster_iter, dopsf=False,
                               normalize=True, invert_function=invert_2d_base,
                               **kwargs) -> (Image, numpy.ndarray):
    """ Predict using image partitions, calling specified predict function

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param image_iterator: Iterator to use for partitioning
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    :return: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]
    """
    
    log.info("invert_with_image_iterator: Inverting by image partitions")
    i = 0
    nchan, npol, _, _ = im.shape
    totalwt = numpy.zeros([nchan, npol])
    for dpatch in image_iterator(im, **kwargs):
        result, sumwt = invert_function(vis, dpatch, dopsf, normalize=False, **kwargs)
        totalwt = sumwt
        # Ensure that we fill in the elements of dpatch instead of creating a new numpy arrray
        dpatch.data[...] = result.data[...]
        assert numpy.max(numpy.abs(dpatch.data)), "Partition image %d appears to be empty" % i
        i += 1
    assert numpy.max(numpy.abs(im.data)), "Output image appears to be empty"
    
    if normalize:
        im = normalize_sumwt(im, totalwt)
    
    return im, totalwt
