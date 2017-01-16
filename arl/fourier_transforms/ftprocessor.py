# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing. These are built on top of the core
functions in arl.fourier_transforms.
"""

import pymp

from scipy.interpolate import griddata

from astropy import units as units
from astropy import wcs
from astropy.constants import c
from astropy.wcs.utils import pixel_to_skycoord

from arl.data.data_models import *
from arl.fourier_transforms.convolutional_gridding import fixed_kernel_grid, \
    fixed_kernel_degrid, weight_gridding, w_beam, anti_aliasing_calculate, anti_aliasing_box
from arl.fourier_transforms.fft_support import fft, ifft, pad_mid, extract_mid
from arl.fourier_transforms.variable_kernels import variable_kernel_grid, variable_kernel_degrid, \
    box_grid, standard_kernel_lambda, w_kernel_lambda
from arl.image.iterators import *
from arl.util.coordinate_support import simulate_point, skycoord_to_lmn
from arl.visibility.iterators import *
from arl.visibility.operations import phaserotate_visibility, create_visibility_from_rows
from arl.util.timing import timing

log = logging.getLogger("fourier_transforms.ftprocessor")


def shift_vis_to_image(vis, im, tangent=True, inverse=False, **kwargs):
    """Shift visibility to the FFT phase centre of the image in place

    :param vis: Visibility data
    :param model: Image model used to determine phase centre
    :returns: visibility with phase shift applied and phasecentre updated
    
    """
    nchan, npol, ny, nx = im.data.shape
    # Convert the FFT definition of the phase center to world coordinates (0 relative)
    image_phasecentre = pixel_to_skycoord(ny // 2, nx // 2, im.wcs)

    if vis.phasecentre.separation(image_phasecentre).value > 1e-15:
        if inverse:
            log.debug("shift_vis_from_image: shifting phasecentre from image phase centre %s to visibility phasecentre "
                      "%s" % (image_phasecentre, vis.phasecentre))
        else:
            log.debug("shift_vis_from_image: shifting phasecentre from vis phasecentre %s to image phasecentre %s" %
                    (vis.phasecentre, image_phasecentre))
        vis = phaserotate_visibility(vis, image_phasecentre, tangent=tangent, inverse=inverse, **kwargs)
        vis.phasecentre = im.phasecentre

    assert type(vis) is Visibility, "after phase_rotation, vis is not a Visibility"
    
    return vis


def get_ftprocessor_params(vis, model, **kwargs):
    """ Common interface to params for predict and invert
    
    :param vis: Visibility data
    :param model: Image model used to determine sampling
    :param padding: Pad images by this factor during processing (2)
    :param kernel: kernel to use {2d|wprojection} (2d)
    :param oversampling: Oversampling factor for convolution function (8)
    :param support: Support of convolution function (width = 2*support+2) (3)
    :returns: nchan, npol, ny, nx, shape, gcf, kernel_type, kernelname, kernel,padding, oversampling, support, cellsize, fov
    """
    padding = get_parameter(kwargs, "padding", 2)
    kernelname = get_parameter(kwargs, "kernel", "2d")
    oversampling = get_parameter(kwargs, "oversampling", 8)
    support = get_parameter(kwargs, "support", 3)
    nchan, npol, ny, nx = model.data.shape
    shape = (padding * ny, padding * nx)
    cellsize = model.wcs.wcs.cdelt[0:2] * numpy.pi / 180.0
    fov = padding * nx * numpy.max(numpy.abs(cellsize))
    uvscale = numpy.outer(cellsize, vis.frequency / c.value)
    assert uvscale[0,0] != 0.0, "Error in uv scaling"

    kernel_type = 'fixed'
    gcf = 1.0
    if kernelname == 'standard-by-row':
        kernel_type = 'variable'
        log.info("get_ftprocessor_params: using calculated spheroidal function by row")
        kernel = standard_kernel_lambda(vis, shape)
    elif kernelname == 'wprojection':
        log.info("get_ftprocessor_params: using wprojection kernel")
        wmax = numpy.max(numpy.abs(vis.w))
        assert wmax > 0, "Maximum w must be > 0.0"
        kernel_type = 'variable'
        r_f = (fov / 2) ** 2 / numpy.abs(cellsize[0])
        log.info("get_ftprocessor_params: Fresnel number = %f" % (r_f))
        delA = get_parameter(kwargs, 'wloss', 0.02)
        # Following equation is from Cornwell, Humphreys, and Voronkov (2012) (equation 24)
        recommended_wstep = numpy.sqrt(2.0 * delA) / (numpy.pi * fov ** 2)
        log.info("get_ftprocessor_params: Recommended wstep = %f" % (recommended_wstep))
        wstep = get_parameter(kwargs, "wstep", recommended_wstep)
        log.info("get_ftprocessor_params: Using w projection with wstep = %f" % (wstep))
        # Now calculate the maximum support for the w kernel
        npixel_kernel = get_parameter(kwargs, "kernelwidth", (int(round(numpy.sin(0.5 * fov) * nx)) // 2))
        log.info("get_ftprocessor_params: w kernel full width = %d pixels" % (npixel_kernel))
        kernel, _ = w_kernel_lambda(vis, shape, fov, wstep=wstep, npixel_kernel=npixel_kernel,
                                    oversampling=oversampling)
        gcf, _ = anti_aliasing_calculate(shape, oversampling)
    elif kernelname == 'box':
        log.info("get_ftprocessor_params: using box car convolution")
        gcf, kernel = anti_aliasing_box(shape)
    else:
        kernelname = '2d'
        gcf, kernel = anti_aliasing_calculate(shape, oversampling)
    
    return nchan, npol, ny, nx, shape, gcf, kernel_type, kernelname, kernel, padding, oversampling, support, \
           cellsize, fov, uvscale



def predict_2d_base(vis, model, **kwargs):
    """ Predict using convolutional degridding.

    This is at the bottom of the layering i.e. all transforms are eventually expressed in terms of
    this function. Any shifting needed is performed here.

    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    nchan, npol, ny, nx, shape, gcf, kernel_type, kernelname, kernel,padding, oversampling, support, cellsize, \
    fov, uvscale = get_ftprocessor_params(vis, model, **kwargs)
 
    uvgrid = fft((pad_mid(model.data, padding * nx) * gcf).astype(dtype=complex))
    
    if kernel_type == 'variable':
        vis.data['vis'] = variable_kernel_degrid(kernel, uvgrid, vis.uvw, uvscale)
    else:
        vis.data['vis'] = fixed_kernel_degrid(kernel, uvgrid, vis.uvw, uvscale)

    # Now we can shift the visibility from the image frame to the original visibility frame
    vis = shift_vis_to_image(vis, model, tangent=True, inverse=True, **kwargs)

    return vis


def predict_2d(vis, model, **kwargs):
    """ Predict using convolutional degridding and w projection
    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_2d: predict using 2d transform")
    return predict_2d_base(vis, model, **kwargs)



def predict_wprojection(vis, model, **kwargs):
    """ Predict using convolutional degridding and w projection
    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_wprojection: predict using wprojection")
    return predict_2d_base(vis, model, kernel='wprojection', **kwargs)



def invert_2d_base(vis, im, dopsf=False, **kwargs):
    """ Invert using 2D convolution function, including w projection
    
    Use the image im as a template. Do PSF in a separate call.

    This is at the bottom of the layering i.e. all transforms are eventually expressed in terms
    of this function. . Any shifting needed is performed here.
    
    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :returns: resulting image
    
    """
    svis = copy.deepcopy(vis)
    
    svis = shift_vis_to_image(svis, im, tangent=True, inverse=False, **kwargs)
    
    nchan, npol, ny, nx, shape, gcf, kernel_type, kernelname, kernel,padding, oversampling, support, cellsize, \
    fov, uvscale = get_ftprocessor_params(vis, im, **kwargs)
    
    # uvw is in metres, v.frequency / c.value converts to wavelengths, the cellsize converts to phase
    # Optionally pad to control aliasing
    imgridpad = numpy.zeros([nchan, npol, padding * ny, padding * nx], dtype='complex')
    uvw = svis.data['uvw']
    if kernel_type == 'variable':
        if dopsf:
            weights = numpy.ones_like(svis.data['vis'])
            imgridpad, sumwt = variable_kernel_grid(kernel, imgridpad, uvw, uvscale, weights,
                                                    svis.data['imaging_weight'])
        else:
            imgridpad, sumwt = variable_kernel_grid(kernel, imgridpad, uvw, uvscale, svis.data['vis'],
                                             svis.data['imaging_weight'])
    else:
        if dopsf:
            weights = numpy.ones_like(svis.data['vis'])
            if kernelname == 'box':
                imgridpad, sumwt = box_grid(kernel, imgridpad, uvw, uvscale, weights, svis.data['imaging_weight'])
            else:
                imgridpad, sumwt = fixed_kernel_grid(kernel, imgridpad, uvw, uvscale, weights,
                                            svis.data['imaging_weight'])
        else:
            if kernelname == 'box':
                imgridpad, sumwt = box_grid(kernel, imgridpad, uvw, uvscale, svis.data['vis'],
                                            svis.data['imaging_weight'])
            else:
                imgridpad, sumwt = fixed_kernel_grid(kernel, imgridpad, uvw, uvscale, svis.data['vis'],
                                            svis.data['imaging_weight'])
    
    imgrid = extract_mid(numpy.real(ifft(imgridpad)) * gcf, npixel=nx)
    
    # Normalise weights for consistency with transform
    sumwt /= float(padding * padding * nx * ny)

    log.debug("invert_2d_base: Peak of unnormalised dirty image = %s" % (numpy.max(imgrid, axis=(2,3))))
    log.debug("invert_2d_base: Sum of gridding weights = %s" % (sumwt))
    if sumwt.any() > 0.0:
        log.debug("invert_2d_base: Peak of normalised dirty image = %s" % (numpy.max(imgrid, axis=(2,3)) / sumwt))

    return create_image_from_array(imgrid, im.wcs), sumwt



def invert_2d(vis, im, dopsf=False, **kwargs):
    """ Invert using prolate spheroidal gridding function

    Use the image im as a template. Do PSF in a separate call.
    
    Note that the image is not normalised but the sum of the weights. This is for ease of use in partitioning.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]

    """
    log.debug("invert_2d: inverting using 2d transform")
    kwargs['kernel'] = '2d'
    return invert_2d_base(vis, im, dopsf, **kwargs)



def invert_wprojection(vis, im, dopsf=False, **kwargs):
    """ Predict using 2D convolution function, including w projection

    Use the image im as a template. Do PSF in a separate call.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]

    """
    log.debug("invert_2d: inverting using wprojection")
    kwargs['kernel'] = 'wprojection'
    return invert_2d_base(vis, im, dopsf, **kwargs)


def fit_uvwplane(vis):
    """ Fit and remove the best fitting plane p u + q v = w

    :param vis: visibility to be fitted
    :returns: direction cosines defining plane
    """
    nvis = len(vis.data)
    before = numpy.std(vis.w)

    su2 = numpy.sum(vis.u * vis.u)
    sv2 = numpy.sum(vis.v * vis.v)
    suv = numpy.sum(vis.u * vis.v)
    suw = numpy.sum(vis.u * vis.w)
    svw = numpy.sum(vis.v * vis.w)
    det = su2 * sv2 - suv ** 2
    p = (sv2 * suw - suv * svw) / det
    q = (su2 * svw - suv * suw) / det
    
    vis.data['uvw'][:, 2] -= p * vis.u + q * vis.v
    after = numpy.max(numpy.std(vis.w))
    log.info('predict_timeslice: Fit to %d rows reduces rms w from %.1f to %.1f wavelengths'
             % (nvis, before, after))
    
    return vis, p, q



def predict_timeslice_serial(vis, model, **kwargs):
    """ Predict using time slices.

    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_timeslice: predicting using time slices")
    nchan, npol, ny, nx, shape, gcf, kernel_type, kernelname, kernel,padding, oversampling, support, cellsize, \
    fov, uvscale = get_ftprocessor_params(vis, model, **kwargs)

    vis.data['vis']*=0.0

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
                             fill_value=0.0).reshape(workimage.data[chan, pol, ...].shape)
                
        # Now we can do the prediction for this slice using a 2d transform
        visslice = predict_2d(visslice, workimage, **kwargs)
        
        vis.data['vis'][rows] += visslice.data['vis']

    return vis


def predict_timeslice(vis, model, **kwargs):
    """ Predict using time slices.

    :param vis: Visibility to be predicted
    :param model: model image
    :param nprocessor: Number of processors to be used (1)
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_timeslice: predicting using time slices")
    
    nproc = get_parameter(kwargs, "nprocessor", 1)
    
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
        
        vis.data.replace_column('vis', shared_vis)

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
    nchan, npol, ny, nx, shape, gcf, kernel_type, kernelname, kernel, padding, oversampling, support, cellsize, \
    fov, uvscale = get_ftprocessor_params(vis, model, **kwargs)
    
    vis.data['vis'] *= 0.0
    
    # Fit and remove best fitting plane for this slice
    vis, p, q = fit_uvwplane(vis)
    
    # Calculate nominal and distorted coordinate systems. We will convert the model
    # from nominal to distorted before predicting.
    lnominal, mnominal, ldistorted, mdistorted = lm_distortion(model, -p, -q)
    workimage = create_image_from_array(copy.deepcopy(model.data), model.wcs)
    log.info("Creating model from SIN projection to oblique SIN projection with params %.6f, %.6f" % (p,q))
    
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
                         fill_value=0.0).reshape(workimage.data[chan, pol, ...].shape)
    
    # Now we can do the prediction for this slice using a 2d transform
    vis = predict_2d(vis, workimage, **kwargs)
    
    return vis


def invert_timeslice(vis, im, dopsf=False, **kwargs):
    """ Invert using time slices (top level function)

    Use the image im as a template. Do PSF in a separate call.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param nprocessor: Number of processors to be used (1)
    :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]

    """
    log.debug("invert_timeslice: inverting using time slices")
    resultimage = create_image_from_array(im.data, im.wcs)
    resultimage.data = pymp.shared.array(resultimage.data.shape)
    resultimage.data *= 0.0
    
    nproc = get_parameter(kwargs, "nprocessor", 1)
    
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
            visslice=create_visibility_from_rows(vis, rows)
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
    nchan, npol, ny, nx, shape, gcf, kernel_type, kernelname, kernel, padding, oversampling, support, cellsize, \
    fov, uvscale = get_ftprocessor_params(vis, im, **kwargs)

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
                         fill_value=0.0).reshape(finalimage.data[chan, pol, ...].shape)

    return finalimage, sumwt



def invert_by_image_partitions(vis, im, image_iterator=raster_iter, dopsf=False,
                               invert_function=invert_2d, **kwargs):
    """ Predict using image partitions, calling specified predict function

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param image_iterator: Iterator to use for partitioning
    :param dopsf: Make the psf instead of the dirty image
    :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]
    """
    
    log.debug("invert_by_image_partitions: Inverting by image partitions")
    i = 0
    nchan, npol, _, _ = im.shape
    totalwt = numpy.zeros([nchan, npol])
    for dpatch in image_iterator(im, **kwargs):
        result, sumwt = invert_function(vis, dpatch, dopsf, **kwargs)
        totalwt += sumwt
        # Ensure that we fill in the elements of dpatch instead of creating a new numpy arrray
        dpatch.data[...] = result.data[...]
        assert numpy.max(numpy.abs(dpatch.data)), "Partition image %d appears to be empty" % i
        i += 1
    assert numpy.max(numpy.abs(im.data)), "Output image appears to be empty"
    
    # Loose thread here: we have to assume that all patchs have the same sumwt
    return im, sumwt



def invert_by_vis_partitions(vis, im, vis_iterator, dopsf=False, invert_function=invert_2d, **kwargs):
    """ Invert using wslices

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :returns: resulting image

    """
    log.debug("invert_by_vis_partitions: Inverting by vis partitions")
    nchan, npol, _, _ = im.shape
    totalwt = numpy.zeros([nchan, npol])
    
    for visslice in vis_iterator(vis, **kwargs):
        result, sumwt = invert_function(visslice, im, dopsf, invert_function, **kwargs)
        totalwt += sumwt
    
    return result, totalwt



def predict_by_image_partitions(vis, model, image_iterator=raster_iter, predict_function=predict_2d,
                                **kwargs):
    """ Predict using image partitions, calling specified predict function

    :param vis: Visibility to be predicted
    :param model: model image
    :param image_iterator: Image iterator used to access the image
    :param predict_function: Function to be used for prediction (allows nesting)
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_by_image_partitions: Predicting by image partitions")
    vis.data['vis']*=0.0
    result = copy.deepcopy(vis)
    for dpatch in image_iterator(model, **kwargs):
        result = predict_function(result, dpatch, **kwargs)
        vis.data['vis'] += result.data['vis']
    return vis



def predict_by_vis_partitions(vis, model, vis_iterator, predict_function=predict_2d, **kwargs):
    """ Predict using vis partitions

    :param vis: Visibility to be predicted
    :param model: model image
    :param vis_iterator: Iterator to use for partitioning
    :param predict_function: Function to be used for prediction (allows nesting)
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_by_vis_partitions: Predicting by vis partitions")
    for vslice in vis_iterator(vis, **kwargs):
        vis.data['vis'] += predict_function(vslice, model, **kwargs).data['vis']
    return vis


def predict_skycomponent_visibility(vis: Visibility, sc: Skycomponent, **kwargs) -> Visibility:
    """Predict the visibility from a Skycomponent, add to existing visibility

    :param vis: Visibility
    :param sc: Skycomponent
    :param spectral_mode: {mfs|channel} (channel)
    :returns: Visibility
    """
    spectral_mode = get_parameter(kwargs, 'spectral_mode', 'channel')
    
    assert_same_chan_pol(vis, sc)
    
    l, m, n = skycoord_to_lmn(sc.direction, vis.phasecentre)
    # The data column has vis:[row,nchan,npol], uvw:[row,3]
    if spectral_mode == 'channel':
        for channel in range(sc.nchan):
            uvw = vis.uvw_lambda(channel)
            phasor = simulate_point(uvw, l, m)
            for pol in range(sc.npol):
                vis.vis[:, channel, pol] += sc.flux[channel, pol] * phasor
    else:
        raise NotImplementedError("mode %s not supported" % spectral_mode)
    
    return vis


def calculate_delta_residual(deltamodel, vis, **kwargs):
    """Calculate the delta in residual for a given delta in model
    
    This calculation does not require the original visibilities.
    """
    return deltamodel



def weight_visibility(vis, im, **kwargs):
    """ Reweight the visibility data using a selected algorithm

    Imaging uses the column "imaging_weight" when imaging. This function sets that column using a
    variety of algorithms
    
    :param vis:
    :param im:
    :returns: visibility with imaging_weights column added and filled
    """
    nchan, npol, ny, nx, shape, gcf, kernel_type, kernelname, kernel,padding, oversampling, support, cellsize, \
    fov, uvscale = get_ftprocessor_params(vis, im, **kwargs)

    # uvw is in metres, v.frequency / c.value converts to wavelengths, the cellsize converts to phase
    density = None
    densitygrid = None
    
    assert nx==im.data.shape[3], "Discrepancy between npixel and size of model image"
    assert ny==im.data.shape[2], "Discrepancy between npixel and size of model image"

    weighting = get_parameter(kwargs, "weighting", "uniform")
    if weighting == 'uniform':
        vis.data['imaging_weight'], density, densitygrid = weight_gridding(im.data.shape, vis.data['uvw'], uvscale,
                                                              vis.data['weight'], weighting)
    elif weighting == 'natural':
        vis.data['imaging_weight'] = vis.data['weight']
    else:
        log.error("Unknown visibility weighting algorithm %s" % weighting)
    
    return vis, density, densitygrid


def create_wcs_from_visibility(vis, **kwargs):
    """Make a world coordinate system from params and Visibility

    :param vis:
    :param imagecentre: Centre of image (SkyCoord)
    :param phasecentre: Phasecentre (Skycoord)
    :param reffrequency: Reference frequency for WCS (Hz)
    :param channelwidth: Channel width (Hz)
    :param cellsize: Cellsize (radians)
    :param npixel: Number of pixels on each axis (512)
    :param frame: Coordinate frame for WCS (ICRS)
    :param equinox: Equinox for WCS (2000.0)
    :returns: WCS
    """
    log.info("create_wcs_from_visibility: Parsing parameters to get definition of WCS")
    imagecentre = get_parameter(kwargs, "imagecentre", vis.phasecentre)
    phasecentre = get_parameter(kwargs, "phasecentre", vis.phasecentre)
    reffrequency = get_parameter(kwargs, "reffrequency", numpy.min(vis.frequency)) * units.Hz
    deffaultbw = vis.frequency[0]
    if len(vis.frequency) > 1:
        deffaultbw = vis.frequency[1] - vis.frequency[0]
    channelwidth = get_parameter(kwargs, "channelwidth", deffaultbw) * units.Hz
    log.info("create_wcs_from_visibility: Defining Image at %s, frequency %s, and bandwidth %s"
             % (imagecentre, reffrequency, channelwidth))
    
    npixel = get_parameter(kwargs, "npixel", 512)
    uvmax = (numpy.abs(vis.data['uvw'][:,0:1]).max() * numpy.max(vis.frequency) / c).value
    log.info("create_wcs_from_visibility: uvmax = %f wavelengths" % uvmax)
    criticalcellsize = 1.0 / (uvmax * 2.0)
    log.info("create_wcs_from_visibility: Critical cellsize = %f radians, %f degrees" % (
        criticalcellsize, criticalcellsize * 180.0 / numpy.pi))
    cellsize = get_parameter(kwargs, "cellsize", 0.5 * criticalcellsize)
    log.info("create_wcs_from_visibility: Cellsize          = %f radians, %f degrees" % (cellsize,
                                                                                         cellsize * 180.0 / numpy.pi))
    if cellsize > criticalcellsize:
        log.info("create_wcs_from_visibility: Resetting cellsize %f radians to criticalcellsize %f radians" % (cellsize, criticalcellsize))
        cellsize = criticalcellsize
    
    npol = get_parameter(kwargs, "npol", vis.data['vis'].shape[2])
    
    # Beware of python indexing order! wcs and the array have opposite ordering
    shape = [len(vis.frequency), npol, npixel, npixel]
    w = wcs.WCS(naxis=4)
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, 1.0, channelwidth.value]
    # The numpy definition of the phase centre of an FFT is n // 2 so that's what we use for
    # the reference pixel but we have to use 1-relative indxing for wcs (arghhh!)
    w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.value, phasecentre.dec.value, 1.0, reffrequency.value]
    w.naxis = 4
    
    w.wcs.radesys = get_parameter(kwargs, 'frame', 'ICRS')
    w.wcs.equinox = get_parameter(kwargs, 'equinox', 2000.0)
    
    return shape, reffrequency, cellsize, w, imagecentre


def create_image_from_visibility(vis, **kwargs):
    """Make an empty imagefrom params and Visibility

    :param vis:
    :returns: WCS
    """
    shape, _, _, w, _ = create_wcs_from_visibility(vis, **kwargs)
    return create_image_from_array(numpy.zeros(shape), wcs=w)


def create_w_term_image(vis, w=None, **kwargs):
    """Create an image with a w term phase term in it
    
    :param vis: Visibility
    :param w: w value to evaluate (calculated as median)
    :returns: Image
    """
    if w is None:
        w = numpy.median(numpy.abs(vis.data['uvw'][:, 2]))
        log.info('create_w_term_image: Creating w term image for median w %f' % w)
    
    im = create_image_from_visibility(vis, **kwargs)
    cellsize = abs(im.wcs.wcs.cdelt[0]) * numpy.pi / 180.0
    _, _, _, npixel = im.data.shape
    im.data = w_beam(npixel, npixel * cellsize, w=w)
    
    fresnel = w * (0.5 * npixel * cellsize) ** 2
    log.info('create_w_term_image: Fresnel number for median w and this field of view and sampling = '
             '%.2f' % (fresnel))
    
    return im
