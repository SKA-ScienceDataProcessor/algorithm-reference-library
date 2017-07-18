""" The wide field imaging equation can be approximated by partitioning the image plane into small regions, treating each separately and then glueing the resulting partitions into one image. We call this image plane partitioning image plane faceting.

.. math::
    V(u,v,w) = \\sum_{i,j} \\frac{1}{\\sqrt{1- l_{i,j}^2- m_{i,j}^2}} e^{-2 \\pi j (ul_{i,j}+um_{i,j} + w(\\sqrt{
    1-l_{i,j}^2-m_{i,j}^2}-1))} \\int  I(\\Delta l, \\Delta m) e^{-2 \\pi j (u\\Delta l_{i,j}+u \\Delta m_{i,j})} dl dm

"""

import numpy

from arl.data.data_models import Visibility, Image
from arl.imaging.base import predict_2d_base, invert_2d_base
from arl.image.iterators import raster_iter
from arl.imaging.iterated import predict_with_image_iterator, invert_with_image_iterator

import logging
log = logging.getLogger(__name__)


def predict_facets(vis: Visibility, model: Image, predict_function=predict_2d_base, **kwargs) -> Visibility:
    """ Predict using image facets, calling specified predict function

    :param vis: Visibility to be predicted
    :param model: model image
    :param predict_function: Function to be used for prediction (allows nesting) (default predict_2d)
    :return: resulting visibility (in place works)
    """
    log.info("predict_facets: Predicting by image facets")
    return predict_with_image_iterator(vis, model, image_iterator=raster_iter, predict_function=predict_function,
                                       **kwargs)


def invert_facets(vis: Visibility, im: Image, dopsf=False, normalize=True, invert_function=invert_2d_base, **kwargs) \
        -> (Image, numpy.ndarray):
    """ Invert using image partitions, calling specified Invert function

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    :param invert_function: Function to be used for inverting (allows nesting) (default invert_2d)
    :return: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]
    """
    
    log.info("invert_facets: Inverting by image facets")
    return invert_with_image_iterator(vis, im, normalize=normalize, image_iterator=raster_iter, dopsf=dopsf,
                                      invert_function=invert_function, **kwargs)
