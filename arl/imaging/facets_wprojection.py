
"""
This is a combination of facets and wprojection. The outer iteration is over facet and the inner is over w.

"""

import numpy

from arl.data.data_models import Visibility, Image
from arl.imaging.wprojection import predict_wprojection, invert_wprojection

from arl.image.iterators import raster_iter
from arl.imaging.iterated import predict_with_image_iterator, invert_with_image_iterator

import logging

log = logging.getLogger(__name__)

def predict_facets_wprojection(vis: Visibility, model: Image, **kwargs)  -> \
        Visibility:
    """ Predict using image facets, calling predict_wprojection

    :param vis: Visibility to be predicted
    :param model: model image
    :return: resulting visibility (in place works)
    """
    log.info("predict_facets_wprojection: Predicting by image facets and w projection")
    return predict_with_image_iterator(vis, model, image_iterator=raster_iter, predict_function=predict_wprojection,
                                **kwargs)

def invert_facets_wprojection(vis: Visibility, im: Image, dopsf=False, normalize=True, **kwargs)  -> (Image,
                                                                                                      numpy.ndarray):
    """ Invert using image partitions, calling invert_wprojection

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    :return: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]
    """
    
    log.info("invert_facets_wprojection: Inverting by image facets and w projection")
    return invert_with_image_iterator(vis, im, normalize=normalize, image_iterator=raster_iter,
                                      invert_function=invert_wprojection, dopsf=dopsf,
                                      **kwargs)