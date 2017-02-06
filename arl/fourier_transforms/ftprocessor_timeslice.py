# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing.
"""
import multiprocessing

import pymp

from scipy.interpolate import griddata
from astropy.coordinates import AltAz, Angle
from astropy.time import Time

from arl.fourier_transforms.ftprocessor_base import *
from arl.image.iterators import *
from arl.image.operations import copy_image, create_empty_image_like, export_image_to_fits, reproject_image
from arl.visibility.iterators import *
from arl.visibility.operations import create_visibility_from_rows

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


def predict_timeslice_serial(vis, model, **kwargs):
    """ Predict using time slices.

    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_timeslice: predicting using time slices")
    nchan, npol, ny, nx = model.data.shape
    
    vis.data['vis'] *= 0.0
    
    for rows in vis_timeslice_iter(vis, **kwargs):
        
        visslice = create_visibility_from_rows(vis, rows)
        
        # Fit and remove best fitting plane for this slice
        visslice, p, q = fit_uvwplane(visslice)
        
        log.info("Creating image from oblique SIN projection with params %.6f, %.6f to SIN projection" % (p, q))
        # Calculate nominal and distorted coordinate systems. We will convert the model
        # from nominal to distorted before predicting.
        lnominal, mnominal, ldistorted, mdistorted = lm_distortion(model, -p, -q)
        workimage = create_empty_image_like(model)
        
        # Use griddata to do the conversion. This could be improved. Only cubic is possible in griddata.
        # The interpolation is ok for invert since the image is smooth but for clean images the
        # interpolation is particularly poor, leading to speckle in the residual image.
        usereproject = get_parameter(kwargs, "usereproject", False)
        if usereproject:
            log.debug('predict_timeslice: Using reproject to convert projection')
            # Set the parameters defining the SIN projection for this plane
            newwcs = model.wcs.deepcopy()
            newwcs.wcs.wcs.set_pv([(0, 0, p), (0, 1, q)])
            # Reproject the model from the natural oblique SIN projection to the non-oblique SIN projection
            workimage, footprintimage = reproject_image(model, newwcs, shape=[nchan, npol, ny, nx])
            workimage.data[footprintimage.data <= 0.0] = 0.0

        else:
            # This is much slower than reproject
            log.debug('predict_timeslice: Using griddata to convert projection')
            for chan in range(nchan):
                for pol in range(npol):
                    workimage.data[chan, pol, ...] = \
                        griddata((mnominal.flatten(), lnominal.flatten()),
                                 values=workimage.data[chan, pol, ...].flatten(),
                                 xi=(mdistorted.flatten(), ldistorted.flatten()),
                                 method='cubic',
                                 fill_value=0.0,
                                 rescale=True).reshape(workimage.data[chan, pol, ...].shape)
        
        # Now we can do the prediction for this slice using a 2d transform
        visslice = predict_2d(visslice, workimage, **kwargs)
        
        vis.data['vis'][rows] += visslice.data['vis']
    
    return vis


def predict_timeslice(vis, model, **kwargs):
    """ Predict using time slices.

    :param vis: Visibility to be predicted
    :param model: model image
    :param timeslice: Timeslice in seconds. If 'auto' will find plausible value
    :param nprocessor: Number of processors to be used (1)
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_timeslice: predicting using time slices")
    
    nproc = get_parameter(kwargs, "nprocessor", 1)
    if nproc == "auto":
        nproc = multiprocessing.cpu_count()
    
    nchan, npol, _, _ = model.data.shape
    
    if nproc > 1:
        
        # Extract the slices and run predict_timeslice_single on each one in parallel
        rowslices = []
        for rows in vis_timeslice_iter(vis, **kwargs):
            rowslices.append(rows)
        nslices = len(rowslices)
        
        log.debug("predict_timeslice: Processing %d time slices %d-way parallel" % (nslices, nproc))
        
        # The visibility column needs to be shared across all processes
        # We have to work around lack of complex data in pymp. For the following trick, see
        # http://stackoverflow.com/questions/2598734/numpy-creating-a-complex-array-from-2-real-ones
        
        shape = vis.data['vis'].shape
        shape = [shape[0], shape[1], shape[2], 2]
        log.debug('Creating shared array of float type and shape %s for visibility' % (str(shape)))
        shared_vis = pymp.shared.array(shape).view(dtype='complex128')[..., 0]
        
        with pymp.Parallel(nproc) as p:
            for slice in p.range(0, nslices):
                rows = rowslices[slice]
                visslice = create_visibility_from_rows(vis, rows)
                visslice = predict_timeslice_single(visslice, model, **kwargs)
                with p.lock:
                    shared_vis[rows] = visslice.data['vis']
        
        vis.data['vis'][...] = shared_vis[...]

    else:
        log.debug("predict_timeslice: Processing time slices serially")
        # Do each slice in turn
        for rows in vis_timeslice_iter(vis, **kwargs):
            visslice = create_visibility_from_rows(vis, rows)
            visslice = predict_timeslice_single(visslice, model, **kwargs)
            vis.data['vis'][rows] += visslice.data['vis']
    
    return vis


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


def invert_timeslice(vis, im, dopsf=False, **kwargs):
    """ Invert using time slices (top level function)

    Use the image im as a template. Do PSF in a separate call.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param timeslice: Timeslice in seconds. If 'auto' will find plausible value
    :param nprocessor: Number of processors to be used (1)
    :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]

    """
    log.debug("invert_timeslice: inverting using time slices")
    resultimage = create_empty_image_like(im)
    resultimage.data = pymp.shared.array(resultimage.data.shape)
    
    nproc = get_parameter(kwargs, "nprocessor", 1)
    if nproc == "auto":
        nproc = multiprocessing.cpu_count()

    inchan, inpol, _, _ = im.data.shape
    
    totalwt = numpy.zeros([inchan, inpol], dtype='float')
    
    if nproc > 1:
        # We need to tell pymp that some arrays are shared
        resultimage.data = pymp.shared.array(resultimage.data.shape)
        resultimage.data *= 0.0
        totalwt = pymp.shared.array([inchan, inpol])
        
        # Extract the slices and run invert_timeslice_single on each one in parallel
        nslices = 0
        rowses = []
        for rows in vis_timeslice_iter(vis, **kwargs):
            nslices += 1
            rowses.append(rows)
        
        log.debug("invert_timeslice: Processing %d time slices %d-way parallel" % (nslices, nproc))
        with pymp.Parallel(nproc) as p:
            for index in p.range(0, nslices):
                visslice = create_visibility_from_rows(vis, rowses[index])
                workimage, sumwt = invert_timeslice_single(visslice, workimage, dopsf, **kwargs)
                resultimage.data += workimage.data
                totalwt += sumwt
    
    else:
        # Do each slice in turn
        i = 0
        for rows in vis_timeslice_iter(vis, **kwargs):
            visslice = create_visibility_from_rows(vis, rows)
            workimage, sumwt = invert_timeslice_single(visslice, im, dopsf, **kwargs)
            resultimage.data += workimage.data
            totalwt += sumwt
            # export_image_to_fits(resultimage, "cumulative_image%d.fits" % (int(numpy.average(visslice.time))))
            # export_image_to_fits(workimage, "corrected_snapshot_image%d.fits" % (int(numpy.average(visslice.time))))
            i+=1
    
    return resultimage, totalwt


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
    
    log.debug("Time=%f, p=%.3f, q=%.3f" % (numpy.average(vis.time), p, q))

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