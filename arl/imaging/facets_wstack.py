
"""
This is a combination of facets and wstacks. The outer iteration is over facet and the inner is over w.

"""
import numpy

from arl.data.data_models import Visibility, Image

from arl.imaging.wstack import predict_wstack, invert_wstack

from arl.image.iterators import raster_iter
from arl.imaging.iterated import predict_with_image_iterator, invert_with_image_iterator

import logging
log = logging.getLogger(__name__)

def predict_facets_wstack(vis: Visibility, model: Image,  **kwargs)  -> Visibility:
    """ Predict using image facets, calling specified predict function

    :param vis: Visibility to be predicted
    :param model: model image
    :return: resulting visibility (in place works)
    """
    log.info("predict_facets_wstack: Predicting by image facets and w stacking")
    return predict_with_image_iterator(vis, model, image_iterator=raster_iter, predict_function=predict_wstack,
                                **kwargs)

def invert_facets_wstack(vis: Visibility, im: Image, dopsf=False, normalize=True, **kwargs)  -> (Image, numpy.ndarray):
    """ Invert using image partitions, calling specified Invert function

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    :return: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]
    """
    
    log.info("invert_facets_wstack: Inverting by image facets and w stacking")
    return invert_with_image_iterator(vis, im, normalize=normalize, image_iterator=raster_iter, dopsf=dopsf,
                                      invert_function=invert_wstack, **kwargs)