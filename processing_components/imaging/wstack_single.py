"""
The w-stacking or w-slicing approach is to partition the visibility data by slices in w. The measurement equation is
approximated as:

.. math::

    V(u,v,w) =\\sum_i \\int \\frac{ I(l,m) e^{-2 \\pi j (w_i(\\sqrt{1-l^2-m^2}-1))})}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm)} dl dm

If images constructed from slices in w are added after applying a w-dependent image plane correction, the w term will be corrected.
"""

import numpy

from data_models.memory_data_models import Visibility, Image, BlockVisibility

from processing_library.image.operations import create_w_term_like

from ..image.operations import copy_image
from ..visibility.base import copy_visibility
from ..visibility.coalesce import convert_blockvisibility_to_visibility, convert_visibility_to_blockvisibility
from ..imaging.base import predict_2d, invert_2d

import logging
log = logging.getLogger(__name__)


def predict_wstack_single(vis, model, remove=True, gcfcf=None, **kwargs) -> Visibility:
    """ Predict using a single w slices.
    
    This processes a single w plane, rotating out the w beam for the average w

    :param vis: Visibility to be predicted
    :param model: model image
    :return: resulting visibility (in place works)
    """

    if not isinstance(vis, Visibility):
        log.debug("predict_wstack_single: Coalescing")
        avis = convert_blockvisibility_to_visibility(vis)
    else:
        avis = vis
        
    avis.data['vis'][...] = 0.0

    log.debug("predict_wstack_single: predicting using single w slice")

    # We might want to do wprojection so we remove the average w
    w_average = numpy.average(avis.w)
    if remove:
     avis.data['uvw'][..., 2] -= w_average
    tempvis = copy_visibility(avis)

    # Calculate w beam and apply to the model. The imaginary part is not needed
    workimage = copy_image(model)
    w_beam = create_w_term_like(model, w_average, avis.phasecentre)
    
    # Do the real part
    workimage.data = w_beam.data.real * model.data
    avis = predict_2d(avis, workimage, gcfcf=gcfcf, **kwargs)
    
    # and now the imaginary part
    workimage.data = w_beam.data.imag * model.data
    tempvis = predict_2d(tempvis, workimage, gcfcf=gcfcf, **kwargs)
    avis.data['vis'] -= 1j * tempvis.data['vis']
    
    if remove:
        avis.data['uvw'][..., 2] += w_average

    if isinstance(vis, BlockVisibility) and isinstance(avis, Visibility):
        return convert_visibility_to_blockvisibility(avis)
    else:
        return avis


def invert_wstack_single(vis: Visibility, im: Image, dopsf, normalize=True, remove=True,
                         gcfcf=None, **kwargs) -> (Image, numpy.ndarray):
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
    if remove:
        vis.data['uvw'][..., 2] -= w_average
    
    reWorkimage, sumwt, imWorkimage = invert_2d(vis, im, dopsf, normalize=normalize, gcfcf=gcfcf, **kwargs)
    
    if remove:
        vis.data['uvw'][..., 2] += w_average

    # Calculate w beam and apply to the model. The imaginary part is not needed
    w_beam = create_w_term_like(im, w_average, vis.phasecentre)
    reWorkimage.data = w_beam.data.real * reWorkimage.data - w_beam.data.imag * imWorkimage.data
    
    return reWorkimage, sumwt
