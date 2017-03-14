# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing.
"""
from scipy.interpolate import griddata

from arl.fourier_transforms.ftprocessor_base import *
from arl.fourier_transforms.ftprocessor_iterated import predict_with_vis_iterator, invert_with_vis_iterator
from arl.image.iterators import *
from arl.image.operations import copy_image, create_empty_image_like, reproject_image
from arl.visibility.iterators import *

log = logging.getLogger(__name__)

def fit_uvwplane_only(vis):
    """ Fit the best fitting plane p u + q v = w

    :param vis: visibility to be fitted
    :returns: direction cosines defining plane
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


def fit_uvwplane(vis):
    """ Fit and remove the best fitting plane p u + q v = w

    :param vis: visibility to be fitted
    :returns: direction cosines defining plane
    """
    nvis = len(vis.data)
    before = numpy.max(numpy.std(vis.w))
    p, q = fit_uvwplane_only(vis)
    vis.data['uvw'][:, 2] -= p * vis.u + q * vis.v
    after = numpy.max(numpy.std(vis.w))
    log.info('predict_timeslice: Fit to %d rows reduces rms w from %.1f to %.1f m'
             % (nvis, before, after))
    return vis, p, q

def invert_timeslice(vis, im, dopsf=False, **kwargs):
    """ Invert using time slices (top level function)

    Use the image im as a template. Do PSF in a separate call.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]

    """
    log.debug("invert_timeslice: inverting using time slices")
    return invert_with_vis_iterator(vis, im, dopsf, vis_iter=vis_timeslice_iter,
                                    invert=invert_timeslice_single, **kwargs)


def predict_timeslice(vis, model, **kwargs):
    """ Predict using time slices.

    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_timeslice: predicting using time slices")

    return predict_with_vis_iterator(vis, model, vis_iter=vis_timeslice_iter,
                                     predict=predict_timeslice_single, **kwargs)


def predict_timeslice_single(vis, model, **kwargs):
    """ Predict using a single time slices.
    
    This fits a single plane and corrects the image geometry.

    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_timeslice: predicting using time slices")
    
    inchan, inpol, ny, nx = model.shape
    
    vis.data['vis'] *= 0.0
    
    # Fit and remove best fitting plane for this slice
    vis, p, q = fit_uvwplane(vis)
    
    # Calculate nominal and distorted coordinate systems. We will convert the model
    # from nominal to distorted before predicting.
    workimage = copy_image(model)
    reprojected_image = copy_image(model)
    log.info("Reprojecting model from SIN projection to oblique SIN projection with params %.6f, %.6f" % (p, q))
    
    # Use griddata to do the conversion. This could be improved. Only cubic is possible in griddata.
    # The interpolation is ok for invert since the image is smooth but for clean images the
    # interpolation is particularly poor, leading to speckle in the residual image.
    usereproject = get_parameter(kwargs, "usereproject", False)
    if usereproject:
        log.debug('predict_timeslice: Using reproject to convert projection')
        # Set the parameters defining the SIN projection for this plane
        newwcs = model.wcs.deepcopy()
        model.wcs.wcs.set_pv([(0, 0, 0.0), (0, 1, 0.0)])
        newwcs.wcs.set_pv([(0, 0, -q), (0, 1, -p)])
        # Reproject the model from the natural SIN projection to the oblique SIN projection
        workimage, footprintimage = reproject_image(model, newwcs, shape=[inchan, inpol, ny, nx])
        workimage.data[footprintimage.data <= 0.0] = 0.0
        # export_image_to_fits(reprojected_image, "reproject%s.fits" % (numpy.average(vis.time)))

    else:
        lnominal, mnominal, ldistorted, mdistorted = lm_distortion(model, -p, -q)
        log.debug('predict_timeslice: Using griddata to convert projection')
        for chan in range(inchan):
            for pol in range(inpol):
                workimage.data[chan, pol, ...] = \
                    griddata((mnominal.flatten(), lnominal.flatten()),
                             values=workimage.data[chan, pol, ...].flatten(),
                             xi=(mdistorted.flatten(), ldistorted.flatten()),
                             method='cubic',
                             fill_value=0.0,
                             rescale=True).reshape(workimage.data[chan, pol, ...].shape)

    
    # Now we can do the prediction for this slice using a 2d transform
    vis = predict_2d(vis, workimage, **kwargs)
    
    return vis

def lm_distortion(im: Image, a, b):
    """Calculate the nominal and distorted coordinates for w=au+bv
    
    :param im: Image with the coordinate system
    :param a, b: parameters in fit
    :returns: meshgrids for l,m nominal and distorted
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


def invert_timeslice_single(vis, im, dopsf, **kwargs):
    """Process single time slice
    
    Extracted for re-use in parallel version
    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    """
    inchan, inpol, ny, nx = im.shape
    
    vis, p, q = fit_uvwplane(vis)
    
    workimage, sumwt = invert_2d(vis, im, dopsf, **kwargs)
    # We don't normalise since that will be done after summing all images
    # export_image_to_fits(workimage, "uncorrected_snapshot_image%d.fits" % (int(numpy.average(vis.time))))

    finalimage = create_empty_image_like(im)
    
    usereproject = get_parameter(kwargs, "usereproject", False)
    if usereproject:
        # Set the parameters defining the SIN projection for this plane
        newwcs = im.wcs.deepcopy()
        newwcs.wcs.set_pv([(0, 0, 0.0), (0, 1, 0.0)])
        workimage.wcs.wcs.set_pv([(0, 0, -p), (0, 1, -q)])
        # Reproject the model from the natural oblique SIN projection to the non-oblique SIN projection
        finalimage, footprintimage = reproject_image(workimage, newwcs, shape=[inchan, inpol, ny, nx])
        finalimage.data[footprintimage.data <= 0.0] = 0.0
    else:
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