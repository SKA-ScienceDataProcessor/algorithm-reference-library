
#
"""
Functions that distributes predict and invert using either just loops or parallel execution
"""

import multiprocessing
import pymp


from arl.fourier_transforms.ftprocessor_params import *
from arl.fourier_transforms.ftprocessor_base import *
from arl.fourier_transforms.ftprocessor_iterated import *
from arl.image.iterators import *
from arl.image.operations import create_empty_image_like
from arl.visibility.iterators import vis_slice_iter
from arl.visibility.operations import create_visibility_from_rows
from arl.data.parameters import get_parameter


log = logging.getLogger(__name__)

def predict_facets(vis, model, predict_function=predict_2d_base, **kwargs):
    """ Predict using image facets, calling specified predict function

    :param vis: Visibility to be predicted
    :param model: model image
    :param predict_function: Function to be used for prediction (allows nesting) (default predict_2d)
    :returns: resulting visibility (in place works)
    """
    log.info("predict_facets: Predicting by image facets")
    return predict_with_image_iterator(vis, model, image_iterator=raster_iter, predict_function=predict_function,
                                **kwargs)

def invert_facets(vis, im, dopsf=False, normalize=True, invert_function=invert_2d_base, **kwargs):
    """ Invert using image partitions, calling specified Invert function

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    :param invert_function: Function to be used for inverting (allows nesting) (default invert_2d)
    :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]
    """
    
    log.info("invert_facets: Inverting by image facets")
    return invert_with_image_iterator(vis, im, normalize=normalize, image_iterator=raster_iter, dopsf=dopsf,
                                      invert_function=invert_function, **kwargs)