"""
The w-stacking or w-slicing approach is to partition the visibility data by slices in w. The measurement equation is 
approximated as:

.. math::

    V(u,v,w) =\\sum_i \\int \\frac{ I(l,m) e^{-2 \\pi j (w_i(\\sqrt{1-l^2-m^2}-1))})}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+um)} dl dm

If images constructed from slices in w are added after applying a w-dependent image plane correction, the w term will be corrected. 
"""

from arl.fourier_transforms.ftprocessor_base import *
from arl.fourier_transforms.ftprocessor_base import *
from arl.fourier_transforms.ftprocessor_iterated import *
from arl.image.iterators import *
from arl.image.operations import copy_image, create_empty_image_like
from arl.visibility.iterators import *
from arl.visibility.operations import create_visibility_from_rows

log = logging.getLogger(__name__)


def predict_wslice(vis, model, **kwargs):
    """ Predict using w slices.

    :param vis: Visibility to be predicted
    :param model: model image
    :param wslice: wslice in seconds. If 'auto' will find plausible value
    :param nprocessor: Number of processors to be used (1)
    :returns: resulting visibility (in place works)
    """
    log.info("predict_wslice: predicting using w slices")

    delA = get_parameter(kwargs, 'wloss', 0.02)
    advice = advise_wide_field(vis, delA)
    kwargs['wslice'] = get_parameter(kwargs, "wstep", advice['w_sampling_primary_beam'])

    return predict_with_vis_iterator(vis, model, vis_iter=vis_wslice_iter,
                                     predict=predict_wslice_single, **kwargs)


def predict_wslice_single(vis, model, **kwargs):
    """ Predict using a single w slices.
    
    This fits a single plane and corrects the image geometry.

    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    if type(vis) is not Visibility:
        avis = coalesce_visibility(vis, **kwargs)
    else:
        avis = vis
        
    avis.data['vis'] *= 0.0
    tempvis = copy_visibility(vis)

    # Calculate w beam and apply to the model. The imaginary part is not needed
    workimage = copy_image(model)
    w_beam = create_w_term_like(model, numpy.average(avis.w))
    
    # Do the real part
    workimage.data = w_beam.data.real * model.data
    vis = predict_2d(vis, workimage, **kwargs)
    
    # and now the imaginary part
    workimage.data = w_beam.data.imag * model.data
    tempvis = predict_2d_base(tempvis, workimage, **kwargs)
    vis.data['vis'] -= 1j * tempvis.data['vis']
    
    return vis


def invert_wslice(vis, im, dopsf=False, normalize=True, **kwargs):
    """ Invert using w slices (top level function)

    Use the image im as a template. Do PSF in a separate call.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    :param wslice: wslice in wavelenngths. If None or omitted will find plausible value
    :param nprocessor: Number of processors to be used (1)
    :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]

    """
    log.info("invert_wslice: inverting using w slices")

    delA = get_parameter(kwargs, 'wloss', 0.02)
    advice = advise_wide_field(vis, delA)
    wslice = get_parameter(kwargs, "wstep", advice['w_sampling_primary_beam'])
    kwargs['wslice'] = wslice

    return invert_with_vis_iterator(vis, im, dopsf, normalize=normalize, vis_iter=vis_wslice_iter,
                                    invert=invert_wslice_single, **kwargs)

def invert_wslice_single(vis, im, dopsf, normalize=True, **kwargs):
    """Process single w slice
    
    Extracted for re-use in parallel version
    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    """
    kwargs['imaginary'] = True
    reWorkimage, sumwt, imWorkimage = invert_2d_base(vis, im, dopsf, normalize=normalize, **kwargs)

    # Calculate w beam and apply to the model. The imaginary part is not needed
    w_beam = create_w_term_like(im, numpy.average(vis.w))
    reWorkimage.data = w_beam.data.real * reWorkimage.data - w_beam.data.imag * imWorkimage.data
    
    return reWorkimage, sumwt