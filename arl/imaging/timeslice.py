"""
The w-term can be viewed as a time-variable distortion. Approximating the array as instantaneously
co-planar, we have that w can be expressed in terms of u,v:

.. math::
    w = a u + b v

Transforming to a new coordinate system:

.. math::

    l' = l + a ( \\sqrt{1-l^2-m^2}-1))

.. math::

    m' = m + b ( \\sqrt{1-l^2-m^2}-1))

Ignoring changes in the normalisation term, we have:

.. math::

    V(u,v,w) =\\int \\frac{ I(l',m')} { \\sqrt{1-l'^2-m'^2}} e^{-2 \\pi j (ul'+um')} dl' dm'


"""
import numpy

from arl.data.data_models import Visibility, Image

from arl.image.operations import copy_image

from arl.imaging.base import predict_2d_base, invert_2d_base
from scipy.interpolate import griddata

from arl.image.operations import create_empty_image_like
from arl.visibility.coalesce import coalesce_visibility

import logging

log = logging.getLogger(__name__)


def fit_uvwplane_only(vis: Visibility) -> (float, float):
    """ Fit the best fitting plane p u + q v = w

    :param vis: visibility to be fitted
    :return: direction cosines defining plane
    """
        
    su2 = numpy.sum(vis.u * vis.u)
    sv2 = numpy.sum(vis.v * vis.v)
    suv = numpy.sum(vis.u * vis.v)
    suw = numpy.sum(vis.u * vis.w)
    svw = numpy.sum(vis.v * vis.w)
    det = su2 * sv2 - suv ** 2
    p = (sv2 * suw - suv * svw) / det
    q = (su2 * svw - suv * suw) / det
    return p, q


def fit_uvwplane(vis: Visibility, remove=False) -> (Image, float, float):
    """ Fit and optionally remove the best fitting plane p u + q v = w

    :param vis: visibility to be fitted
    :param remove: Remove the fitted w permanently from vis?
    :return: direction cosines defining plane
    """
    nvis = len(vis.data)
    before = numpy.max(numpy.std(vis.w))
    p, q = fit_uvwplane_only(vis)
    residual = vis.data['uvw'][:, 2] - (p * vis.u + q * vis.v)
    after = numpy.max(numpy.std(residual))
    log.debug('fit_uvwplane: Fit to %d rows reduces rms w from %.1f to %.1f m'
              % (nvis, before, after))
    if remove:
        vis.data['uvw'][:, 2] -= p * vis.u + q * vis.v
    return vis, p, q


def predict_timeslice_single(vis: Visibility, model: Image, predict=predict_2d_base, remove=True,
                             **kwargs) -> Visibility:
    """ Predict using a single time slices.
    
    This fits a single plane and corrects the image geometry.

    :param vis: Visibility to be predicted
    :param model: model image
    :param predict:
    :param remove: Remove fitted w (so that wprojection will do the right thing)
    :return: resulting visibility (in place works)
    """
    log.debug("predict_timeslice: predicting using time slices")

    inchan, inpol, ny, nx = model.shape
    
    vis.data['vis'] *= 0.0
    
    if not isinstance(vis, Visibility):
        avis = coalesce_visibility(vis, **kwargs)
    else:
        avis = vis

    # Fit and remove best fitting plane for this slice
    avis, p, q = fit_uvwplane(avis, remove=remove)
    
    # Calculate nominal and distorted coordinate systems. We will convert the model
    # from nominal to distorted before predicting.
    workimage = copy_image(model)
    
    # Use griddata to do the conversion. This could be improved. Only cubic is possible in griddata.
    # The interpolation is ok for invert since the image is smooth but for clean images the
    # interpolation is particularly poor, leading to speckle in the residual image.
    lnominal, mnominal, ldistorted, mdistorted = lm_distortion(model, -p, -q)
    for chan in range(inchan):
        for pol in range(inpol):
            workimage.data[chan, pol, ...] = \
                griddata((mnominal.flatten(), lnominal.flatten()),
                         values=workimage.data[chan, pol, ...].flatten(),
                         xi=(mdistorted.flatten(), ldistorted.flatten()),
                         method='cubic',
                         fill_value=0.0,
                         rescale=True).reshape(workimage.data[chan, pol, ...].shape)

    avis = predict(avis, workimage, **kwargs)
    
    return avis


def lm_distortion(im: Image, a, b) -> (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray):
    """Calculate the nominal and distorted coordinates for w=au+bv
    
    :param im: Image with the coordinate system
    :param a, b: parameters in fit
    :return: meshgrids for l,m nominal and distorted
    """
    ny = im.shape[2]
    nx = im.shape[3]
    cy = im.wcs.wcs.crpix[1] - 1
    cx = im.wcs.wcs.crpix[0] - 1
    dy = im.wcs.wcs.cdelt[1] * (numpy.pi / 180.0)
    dx = im.wcs.wcs.cdelt[0] * (numpy.pi / 180.0)
    
    lnominal1d = (numpy.arange(0, nx) - cx) * dx
    mnominal1d = (numpy.arange(0, ny) - cy) * dy
    
    l2d, m2d = numpy.meshgrid(lnominal1d, mnominal1d)
    
    dn2d = numpy.sqrt(1.0 - (l2d * l2d + m2d * m2d)) - 1.0
    
    ldistorted = l2d + a * dn2d
    mdistorted = m2d + b * dn2d

    return l2d, m2d, ldistorted, mdistorted


def invert_timeslice_single(vis: Visibility, im: Image, dopsf, normalize=True,
                            **kwargs) -> (Image, numpy.ndarray):
    """Process single time slice
    
    Extracted for re-use in parallel version
    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    """
    inchan, inpol, ny, nx = im.shape

    if not isinstance(vis, Visibility):
        avis = coalesce_visibility(vis, **kwargs)
    else:
        avis = vis

    log.debug("invert_timeslice: inverting using time slices")

    avis, p, q = fit_uvwplane(avis, remove=True)
    
    workimage, sumwt = invert_2d_base(avis, im, dopsf, normalize=normalize, **kwargs)

    finalimage = create_empty_image_like(im)
    
    # Use griddata to do the conversion. This could be improved. Only cubic is possible in griddata.
    # The interpolation is ok for invert since the image is smooth.
    
    # Calculate nominal and distorted coordinates. The image is in distorted coordinates so we
    # need to convert back to nominal
    lnominal, mnominal, ldistorted, mdistorted = lm_distortion(workimage, -p, -q)

    for chan in range(inchan):
        for pol in range(inpol):
            finalimage.data[chan, pol, ...] = \
                griddata((mdistorted.flatten(), ldistorted.flatten()),
                         values=workimage.data[chan, pol, ...].flatten(),
                         method='cubic',
                         xi=(mnominal.flatten(), lnominal.flatten()),
                         fill_value=0.0,
                         rescale=True).reshape(finalimage.data[chan, pol, ...].shape)
    
    return finalimage, sumwt
