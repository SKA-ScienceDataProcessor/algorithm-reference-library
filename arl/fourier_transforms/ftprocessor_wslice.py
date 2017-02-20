# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing.
"""
import multiprocessing

import pymp

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
    log.debug("predict_wslice: predicting using w slices")

    return predict_with_vis_iterator(vis, model, vis_iter=vis_wslice_iter,
                                     predict=predict_wslice_single, **kwargs)


def predict_wslice_single(vis, model, **kwargs):
    """ Predict using a single w slices.
    
    This fits a single plane and corrects the image geometry.

    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_wslice: predicting using w slices")
    
    vis.data['vis'] *= 0.0
    tempvis = copy_visibility(vis)

    # Calculate w beam and apply to the model. The imaginary part is not needed
    workimage = copy_image(model)
    w_beam = create_w_term_like(model, numpy.average(vis.w))
    
    # Do the real part
    workimage.data = w_beam.data.real * model.data
    vis = predict_2d(vis, workimage, **kwargs)
    
    # and now the imaginary part
    workimage.data = w_beam.data.imag * model.data
    tempvis = predict_2d_base(tempvis, workimage, **kwargs)
    vis.data['vis'] -= 1j * tempvis.data['vis']
    
    return vis


def invert_wslice(vis, im, dopsf=False, **kwargs):
    """ Invert using w slices (top level function)

    Use the image im as a template. Do PSF in a separate call.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param wslice: wslice in seconds. If 'auto' will find plausible value
    :param nprocessor: Number of processors to be used (1)
    :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]

    """
    log.debug("invert_wslice: inverting using w slices")
    return invert_with_vis_iterator(vis, im, dopsf, vis_iter=vis_wslice_iter,
                                    invert=invert_wslice_single, **kwargs)

def invert_wslice_single(vis, im, dopsf, **kwargs):
    """Process single w slice
    
    Extracted for re-use in parallel version
    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    """
    kwargs['imaginary'] = True
    reWorkimage, sumwt, imWorkimage = invert_2d_base(vis, im, dopsf, **kwargs)
    # We don't normalise since that will be done after summing all images
    # export_image_to_fits(workimage, "uncorrected_snapshot_image%d.fits" % (int(numpy.average(vis.w))))

    # Calculate w beam and apply to the model. The imaginary part is not needed
    w_beam = create_w_term_like(im, numpy.average(vis.w))
    reWorkimage.data = w_beam.data.real * reWorkimage.data - w_beam.data.imag * imWorkimage.data
    
    return reWorkimage, sumwt