"""
This is a combination of wprojection and wstacks. The explicit iteration is over wplanes. Wprojection is just
an option in the normal gridding for each plane.

"""

from arl.imaging.iterated import *
from arl.imaging.wstack import *

from arl.image.iterators import *
from arl.visibility.iterators import *
from arl.imaging.base import *

log = logging.getLogger(__name__)


def predict_wprojection_wstack(vis: Visibility, model: Image, **kwargs) -> Visibility:
    """ Predict using convolutional degridding with w projection and wstacking

    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_wprojection: predict using wstack and wprojection")
    kwargs['kernel'] = 'wprojection'
    return predict_wstack(vis, model, **kwargs)


def invert_wprojection_wstack(vis: Visibility, im: Image, dopsf=False, normalize=True, **kwargs) -> (Image, numpy.ndarray):
    """ Predict using 2D convolution function, including w projection and stacking

    Use the image im as a template. Do PSF in a separate call.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]

    """
    log.info("invert_2d: inverting using wstack and wprojection")
    kwargs['kernel'] = "wprojection"
    return invert_wstack(vis, im, dopsf, normalize=normalize, **kwargs)

