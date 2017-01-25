# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing. These are built on top of the core
functions in arl.fourier_transforms.
"""
import multiprocessing

import pymp

from scipy.interpolate import griddata

from arl.fourier_transforms.ftprocessor_base import *
from arl.image.iterators import *
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
    nchan, npol, _, _ = model.data.shape
    
    vis.data['vis'] *= 0.0
    
    for rows in vis_timeslice_iter(vis, **kwargs):
        
        visslice = create_visibility_from_rows(vis, rows)
        
        # Fit and remove best fitting plane for this slice
        visslice, p, q = fit_uvwplane(visslice)
        
        log.info("Creating image from oblique SIN projection with params %.6f, %.6f to SIN projection" % (p, q))
        # Calculate nominal and distorted coordinate systems. We will convert the model
        # from nominal to distorted before predicting.
        lnominal, mnominal, ldistorted, mdistorted = lm_distortion(model, -p, -q)
        workimage = create_image_from_array(copy.deepcopy(model.data), model.wcs)
        
        # Use griddata to do the conversion. This could be improved. Only cubic is possible in griddata.
        # The interpolation is ok for invert since the image is smooth but for clean images the
        # interpolation is particularly poor, leading to speckle in the residual image.
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
    """ Predict using time slices.

    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_timeslice: predicting using time slices")
    
    nchan, npol, ny, nx = model.shape
    
    vis.data['vis'] *= 0.0
    
    # Fit and remove best fitting plane for this slice
    vis, p, q = fit_uvwplane(vis)
    
    # Calculate nominal and distorted coordinate systems. We will convert the model
    # from nominal to distorted before predicting.
    lnominal, mnominal, ldistorted, mdistorted = lm_distortion(model, -p, -q)
    workimage = create_image_from_array(copy.deepcopy(model.data), model.wcs)
    log.info("Reprojecting model from SIN projection to oblique SIN projection with params %.6f, %.6f" % (p, q))
    
    # Use griddata to do the conversion. This could be improved. Only cubic is possible in griddata.
    # The interpolation is ok for invert since the image is smooth but for clean images the
    # interpolation is particularly poor, leading to speckle in the residual image.
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
    resultimage = create_image_from_array(im.data, im.wcs)
    resultimage.data = pymp.shared.array(resultimage.data.shape)
    resultimage.data *= 0.0
    
    nproc = get_parameter(kwargs, "nprocessor", 1)
    if nproc == "auto":
        nproc = multiprocessing.cpu_count()

    nchan, npol, _, _ = im.data.shape
    
    totalwt = numpy.zeros([nchan, npol], dtype='float')
    
    if nproc > 1:
        # We need to tell pymp that some arrays are shared
        resultimage.data = pymp.shared.array(resultimage.data.shape)
        resultimage.data *= 0.0
        totalwt = pymp.shared.array([nchan, npol])
        
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
                workimage, sumwt = invert_timeslice_single(visslice, im, dopsf, **kwargs)
                resultimage.data += workimage.data
                totalwt += sumwt
    
    else:
        # Do each slice in turn
        for rows in vis_timeslice_iter(vis, **kwargs):
            visslice = create_visibility_from_rows(vis, rows)
            workimage, sumwt = invert_timeslice_single(visslice, im, dopsf, **kwargs)
            resultimage.data += workimage.data
            totalwt += sumwt
    
    return resultimage, totalwt


def lm_distortion(im, a, b):
    ny = im.shape[2]
    nx = im.shape[3]
    cy = im.wcs.wcs.crpix[1]
    cx = im.wcs.wcs.crpix[0]
    dy = im.wcs.wcs.cdelt[1] * numpy.pi / 180.0
    dx = im.wcs.wcs.cdelt[0] * numpy.pi / 180.0
    
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
    nchan, npol, ny, nx = im.shape
    
    vis, p, q = fit_uvwplane(vis)
    
    workimage, sumwt = invert_2d(vis, im, dopsf, **kwargs)
    
    # Calculate nominal and distorted coordinates. The image is in distorted coordinates so we
    # need to convert back to nominal
    lnominal, mnominal, ldistorted, mdistorted = lm_distortion(workimage, -p, -q)
    finalimage = create_image_from_array(im.data, im.wcs)
    
    # Use griddata to do the conversion. This could be improved. Only cubic is possible in griddata.
    # The interpolation is ok for invert since the image is smooth.
    for chan in range(nchan):
        for pol in range(npol):
            finalimage.data[chan, pol, ...] = \
                griddata((mdistorted.flatten(), ldistorted.flatten()),
                         values=workimage.data[chan, pol, ...].flatten(),
                         method='cubic',
                         xi=(mnominal.flatten(), lnominal.flatten()),
                         fill_value=0.0,
                         rescale=True).reshape(finalimage.data[chan, pol, ...].shape)
    
    return finalimage, sumwt