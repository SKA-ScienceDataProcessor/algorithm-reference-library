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
import logging

log = logging.getLogger(__name__)

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

from data_models.memory_data_models import Visibility, Image

from processing_components.image.operations import reproject_image

from processing_components.imaging.base import predict_2d, invert_2d


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
    before = numpy.max(numpy.abs(vis.w))
    p, q = fit_uvwplane_only(vis)
    residual = vis.data['uvw'][:, 2] - (p * vis.u + q * vis.v)
    after = numpy.max(numpy.abs(residual))
    if numpy.abs(p) > 1e-7 or numpy.abs(q) > 1e-7:
        log.debug('fit_uvwplane: Fit to %d rows reduces max abs w from %.1f to %.1f m'
                % (nvis, before, after))
    if remove:
        vis.data['uvw'][:, 2] -= p * vis.u + q * vis.v
    return vis, p, q


def predict_timeslice_single(vis: Visibility, model: Image, predict=predict_2d, remove=True,
                             gcfcf=None, **kwargs) -> Visibility:
    """ Predict using a single time slices.

    This fits a single plane and corrects the image geometry.

    :param vis: Visibility to be predicted
    :param model: model image
    :param predict:
    :param remove: Remove fitted w (so that wprojection will do the right thing)
    :param gcfcf: (Grid correction function, convolution function)
    :return: resulting visibility (in place works)
    """
    
    assert isinstance(vis, Visibility), vis
    
    vis.data['vis'][...] = 0.0
    
    # Fit and remove best fitting plane for this slice
    uvw = vis.uvw
    avis, p, q = fit_uvwplane(vis, remove=remove)
    
    # We want to describe work image as distorted. We describe the distortion by putting
    # the olbiquity parameters in the wcs. The input model should be described as having
    # zero olbiquity parameters.
    # Note that this has to be zero relative in first element, one relative in second!!!
    if numpy.abs(p) > 1e-7 or numpy.abs(q) > 1e-7:

        newwcs = model.wcs.deepcopy()
        newwcs.wcs.set_pv([(0, 1, -p), (0, 2, -q)])
        workimage, footprintimage = reproject_image(model, newwcs, shape=model.shape)
        workimage.data[footprintimage.data <= 0.0] = 0.0
        workimage.wcs.wcs.set_pv([(0, 1, -p), (0, 2, -q)])
    
        # Now we can do the predict
        vis = predict(avis, workimage, gcfcf=gcfcf, **kwargs)
    else:
        vis = predict(avis, model, gcfcf=gcfcf, **kwargs)
    
    if remove:
        avis.data['uvw'][...] = uvw

    return vis


def invert_timeslice_single(vis: Visibility, im: Image, dopsf, normalize=True, remove=True,
                            gcfcf=None, **kwargs) -> (Image, numpy.ndarray):
    """Process single time slice

    Extracted for re-use in parallel version
    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param gcfcf: (Grid correction function, convolution function)
    :param normalize: Normalize by the sum of weights (True)
    """
    assert isinstance(vis, Visibility), vis
    
    uvw = vis.uvw
    vis, p, q = fit_uvwplane(vis, remove=remove)
    
    workimage, sumwt = invert_2d(vis, im, dopsf, normalize=normalize, gcfcf=gcfcf, **kwargs)
    # Work image is distorted. We describe the distortion by putting the olbiquity parameters in
    # the wcs. The output image should be described as having zero olbiquity parameters.
    
    if numpy.abs(p) > 1e-7 or numpy.abs(q) > 1e-7:
        # Note that this has to be zero relative in first element, one relative in second!!!!
        workimage.wcs.wcs.set_pv([(0, 1, -p), (0, 2, -q)])
    
        finalimage, footprint = reproject_image(workimage, im.wcs, im.shape)
        finalimage.data[footprint.data <= 0.0] = 0.0
        finalimage.wcs.wcs.set_pv([(0, 1, 0.0), (0, 2, 0.0)])

        if remove:
            vis.data['uvw'][...] = uvw

        return finalimage, sumwt
    else:
        if remove:
            vis.data['uvw'][...] = uvw
    
        return workimage, sumwt
