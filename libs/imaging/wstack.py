"""
The w-stacking or w-slicing approach is to partition the visibility data by slices in w. The measurement equation is
approximated as:

.. math::

    V(u,v,w) =\\sum_i \\int \\frac{ I(l,m) e^{-2 \\pi j (w_i(\\sqrt{1-l^2-m^2}-1))})}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm)} dl dm

If images constructed from slices in w are added after applying a w-dependent image plane correction, the w term will be corrected.
"""

import numpy

from libs.data.data_models import Visibility, Image, BlockVisibility

from libs.image.operations import copy_image
from libs.visibility.base import copy_visibility
from libs.visibility.iterators import vis_wslice_iter
from libs.visibility.coalesce import coalesce_visibility, decoalesce_visibility
from libs.imaging.base import predict_2d_base, invert_2d_base
from libs.image.operations import create_w_term_like

import logging
log = logging.getLogger(__name__)


def predict_wstack_single(vis, model, remove=True, facets=1, vis_slices=1, **kwargs) -> Visibility:
    """ Predict using a single w slices.
    
    This processes a single w plane, rotating out the w beam for the average w

    :param vis: Visibility to be predicted
    :param model: model image
    :return: resulting visibility (in place works)
    """

    if not isinstance(vis, Visibility):
        log.debug("predict_wstack_single: Coalescing")
        avis = coalesce_visibility(vis, **kwargs)
    else:
        avis = vis
        
    log.debug("predict_wstack_single: predicting using single w slice")

    avis.data['vis'] *= 0.0
    # We might want to do wprojection so we remove the average w
    w_average = numpy.average(avis.w)
    avis.data['uvw'][..., 2] -= w_average
    tempvis = copy_visibility(avis)

    # Calculate w beam and apply to the model. The imaginary part is not needed
    workimage = copy_image(model)
    w_beam = create_w_term_like(model, w_average, vis.phasecentre)
    
    # Do the real part
    workimage.data = w_beam.data.real * model.data
    avis = predict_2d_base(avis, workimage, facets=1, vis_slices=1, **kwargs)
    
    # and now the imaginary part
    workimage.data = w_beam.data.imag * model.data
    tempvis = predict_2d_base(tempvis, workimage, facets=facets, vis_slices=vis_slices, **kwargs)
    avis.data['vis'] -= 1j * tempvis.data['vis']
    
    if not remove:
        avis.data['uvw'][..., 2] += w_average

    if isinstance(vis, BlockVisibility) and isinstance(avis, Visibility):
        log.debug("imaging.predict decoalescing post prediction")
        return decoalesce_visibility(avis)
    else:
        return avis


def invert_wstack_single(vis: Visibility, im: Image, dopsf, normalize=True, remove=True, facets=1, vis_slices=1,
                         **kwargs) -> (Image, numpy.ndarray):
    """Process single w slice
    
    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    """
    log.debug("invert_wstack_single: predicting using single w slice")
    
    kwargs['imaginary'] = True
    
    assert isinstance(vis, Visibility), vis
    
    # We might want to do wprojection so we remove the average w
    w_average = numpy.average(vis.w)
    vis.data['uvw'][..., 2] -= w_average
    
    reWorkimage, sumwt, imWorkimage = invert_2d_base(vis, im, dopsf, normalize=normalize, facets=facets,
                                                     vis_slices=vis_slices, **kwargs)
    
    if not remove:
        vis.data['uvw'][..., 2] += w_average

    # Calculate w beam and apply to the model. The imaginary part is not needed
    w_beam = create_w_term_like(im, w_average, vis.phasecentre)
    reWorkimage.data = w_beam.data.real * reWorkimage.data - w_beam.data.imag * imWorkimage.data
    
    return reWorkimage, sumwt
