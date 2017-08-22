"""
The w-stacking or w-slicing approach is to partition the visibility data by slices in w. The measurement equation is 
approximated as:

.. math::

    V(u,v,w) =\\sum_i \\int \\frac{ I(l,m) e^{-2 \\pi j (w_i(\\sqrt{1-l^2-m^2}-1))})}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm)} dl dm

If images constructed from slices in w are added after applying a w-dependent image plane correction, the w term will be corrected. 
"""

import numpy


from arl.data.data_models import Visibility, Image
from arl.imaging.iterated import predict_with_vis_iterator, invert_with_vis_iterator

from arl.image.operations import copy_image
from arl.visibility.base import copy_visibility
from arl.visibility.iterators import vis_wstack_iter
from arl.visibility.coalesce import coalesce_visibility
from arl.imaging.base import predict_2d_base, invert_2d_base
from arl.image.operations import create_w_term_like

import logging
log = logging.getLogger(__name__)


def predict_wstack(vis: Visibility, model: Image, **kwargs) -> Visibility:
    """ Predict using w stacking.

    Note that wprojection can be performed inside wstacking by e.g. ::
    
        vis = predict_wstack(vis, im, vis_slices=10, wstep=2.0, kernel='wprojection')


    :param vis: Visibility to be predicted
    :param vis_slices: Number of slices in the wstack
    :param wstack: size of stack slice in wavelengths (used in vis_slices is not set)
    :param model: model image
    :return: resulting visibility (in place works)
    """
    log.info("predict_wstack: predicting using wstack")

    return predict_with_vis_iterator(vis, model, vis_iter=vis_wstack_iter, predict=predict_wstack_single, **kwargs)


def predict_wstack_single(vis, model, predict_inner=predict_2d_base, **kwargs) -> Visibility:
    """ Predict using a single w slices.
    
    This processes a single w plane, rotating out the w beam for the average w

    :param vis: Visibility to be predicted
    :param model: model image
    :return: resulting visibility (in place works)
    """

    if type(vis) is not Visibility:
        avis = coalesce_visibility(vis, **kwargs)
    else:
        avis = vis
        
    log.debug("predict_wstack_single: predicting using single w slice")

    avis.data['vis'] *= 0.0
    # We might want to do wprojection so we remove the average w
    w_average = numpy.average(avis.w)
    avis.data['uvw'][...,2] -= w_average
    tempvis = copy_visibility(avis)

    # Calculate w beam and apply to the model. The imaginary part is not needed
    workimage = copy_image(model)
    w_beam = create_w_term_like(model, w_average, vis.phasecentre)
    
    # Do the real part
    workimage.data = w_beam.data.real * model.data
    avis = predict_inner(avis, workimage, **kwargs)
    
    # and now the imaginary part
    workimage.data = w_beam.data.imag * model.data
    tempvis = predict_inner(tempvis, workimage, **kwargs)
    avis.data['vis'] -= 1j * tempvis.data['vis']
    
    avis.data['uvw'][...,2] += w_average

    return avis

def invert_wstack(vis: Visibility, im: Image, dopsf=False, normalize=True, **kwargs) -> (Image, numpy.ndarray):
    """ Invert using w stacking

    Use the image im as a template. Do PSF in a separate call.
    
    Note that wprojection can be performed inside wstacking by e.g. ::
    
        dirty = invert_wstack(vis, im, vis_slices=10, wstep=2.0, kernel='wprojection')


    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    :param vis_slices: Number of slices in the wstack
    :return: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]

    """
    log.info("invert_wstack: inverting using wstack")

    return invert_with_vis_iterator(vis, im, dopsf, normalize=normalize, vis_iter=vis_wstack_iter,
                                    invert=invert_wstack_single, **kwargs)

def invert_wstack_single(vis: Visibility, im: Image, dopsf, normalize=True, invert_inner=invert_2d_base, **kwargs) -> (Image, numpy.ndarray):
    """Process single w slice
    
    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    """
    log.debug("invert_wstack_single: predicting using single w slice")
    
    kwargs['imaginary'] = True
    
    assert type(vis) == Visibility
    
    # We might want to do wprojection so we remove the average w
    w_average = numpy.average(vis.w)
    vis.data['uvw'][...,2] -= w_average
    reWorkimage, sumwt, imWorkimage = invert_inner(vis, im, dopsf, normalize=normalize, **kwargs)
    vis.data['uvw'][...,2] += w_average

    # Calculate w beam and apply to the model. The imaginary part is not needed
    w_beam = create_w_term_like(im, w_average, vis.phasecentre)
    reWorkimage.data = w_beam.data.real * reWorkimage.data - w_beam.data.imag * imWorkimage.data
    
    return reWorkimage, sumwt
