"""
The w-stacking or w-slicing approach is to partition the visibility data by slices in w. The measurement equation is 
approximated as:

.. math::

    V(u,v,w) =\\sum_i \\int \\frac{ I(l,m) e^{-2 \\pi j (w_i(\\sqrt{1-l^2-m^2}-1))})}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm)} dl dm

If images constructed from slices in w are added after applying a w-dependent image plane correction, the w term will be corrected. 
"""

from arl.imaging.iterated import *
from arl.imaging.wstack import *

from arl.image.iterators import *
from arl.visibility.iterators import *
from arl.imaging.base import *

log = logging.getLogger(__name__)


def predict_wprojection_wstack(vis: Visibility, model: Image, **kwargs) -> Visibility:
    """ Predict using convolutional degridding with w projection and wstacking

    For a fixed w, the measurement equation can be stated as as a convolution in Fourier space.

    .. math::

        V(u,v,w) =G_w(u,v) \\ast \\int \\frac{I(l,m)}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm)} dl dm$$

    where the convolution function is:

    .. math::

        G_w(u,v) = \\int \\frac{1}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm + w(\\sqrt{1-l^2-m^2}-1))} dl dm


    Hence when degridding, we can use the transform of the w beam to correct this effect.

    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_wprojection: predict using wstack and wprojection")
    kwargs['kernel'] = 'wprojection'
    return predict_wstack(vis, model, **kwargs)


def invert_wprojection_wstack(vis: Visibility, im: Image, dopsf=False, normalize=True, **kwargs) -> (Image, numpy.ndarray):
    """ Predict using 2D convolution function, including w projection and stacking

    For a fixed w, the measurement equation can be stated as as a convolution in Fourier space.

    .. math::

        V(u,v,w) =G_w(u,v) \\ast \\int \\frac{I(l,m)}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm)} dl dm$$

    where the convolution function is:

    .. math::

        G_w(u,v) = \\int \\frac{1}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm + w(\\sqrt{1-l^2-m^2}-1))} dl dm


    Hence when degridding, we can use the transform of the w beam to correct this effect.

    Use the image im as a template. Do PSF in a separate call.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]

    """
    log.info("invert_2d: inverting using wstack and wprojection")
    kwargs['kernel'] = "wprojection"
    return invert_wstack(vis, im, dopsf, normalize=normalize, **kwargs)

