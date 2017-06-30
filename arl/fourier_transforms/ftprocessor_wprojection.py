"""
W projection functions

    For a fixed w, the measurement equation can be stated as as a convolution in Fourier space.
    
    .. math::

        V(u,v,w) =G_w(u,v) \\ast \\int \\frac{I(l,m)}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+um)} dl dm$$

    where the convolution function is:
    
    .. math::

        G_w(u,v) = \\int \\frac{1}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+um + w(\\sqrt{1-l^2-m^2}-1))} dl dm

"""

from arl.fourier_transforms.ftprocessor_base import predict_2d_base, invert_2d_base
from arl.image.iterators import *

log = logging.getLogger(__name__)



def predict_wprojection(vis, model, **kwargs):
    """ Predict using convolutional degridding and w projection.
    
    For a fixed w, the measurement equation can be stated as as a convolution in Fourier space. 
    
    .. math::

        V(u,v,w) =G_w(u,v) \\ast \\int \\frac{I(l,m)}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+um)} dl dm$$

    where the convolution function is:
    
    .. math::

        G_w(u,v) = \\int \\frac{1}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+um + w(\\sqrt{1-l^2-m^2}-1))} dl dm


    Hence when degridding, we can use the transform of the w beam to correct this effect.
    
    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_wprojection: predict using wprojection")
    return predict_2d_base(vis, model, kernel='wprojection', **kwargs)

def invert_wprojection(vis, im, dopsf=False, normalize=True, **kwargs):
    """ Predict using 2D convolution function, including w projection

    For a fixed w, the measurement equation can be stated as as a convolution in Fourier space.
    
    .. math::

        V(u,v,w) =G_w(u,v) \\ast \\int \\frac{I(l,m)}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+um)} dl dm$$

    where the convolution function is:
    
    .. math::

        G_w(u,v) = \\int \\frac{1}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+um + w(\\sqrt{1-l^2-m^2}-1))} dl dm


    Hence when degridding, we can use the transform of the w beam to correct this effect.
    
    Use the image im as a template. Do PSF in a separate call.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]

    """
    log.info("invert_2d: inverting using wprojection")
    kwargs['kernel'] = "wprojection"
    return invert_2d_base(vis, im, dopsf, normalize=normalize, **kwargs)
