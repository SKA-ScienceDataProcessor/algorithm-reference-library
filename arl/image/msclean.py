# Tim Cornwell <realtimcornwell@gmail.com>
""" Image Deconvolution functions

"""
import numpy

from arl.image.hogbom import argmax, overlapIndices
import logging

log = logging.getLogger(__name__)


def msclean(dirty, psf, window, gain, thresh, niter, scales, fracthresh):
    """ Perform multiscale clean

    Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc, 2008 vol. 2 pp. 793-801)
    
    This version operates on numpy arrays.

    :param fracthresh:
    :param dirty: The dirty image, i.e., the image to be deconvolved
    :param psf: The point spread-function
    :param window: Regions where clean components are allowed. If
    True, then all of the dirty image is assumed to be allowed for
    clean components
    :param gain: The "loop gain", i.e., the fraction of the brightest
    pixel that is removed in each iteration
    :param thresh: Cleaning stops when the maximum of the absolute
    deviation of the residual is less than this value
    :param niter: Maximum number of components to make if the
    threshold "thresh" is not hit
    :param scales: Scales (in pixels width) to be used
    :param fracthres: Fractional stopping threshold
    :returns: clean component image, residual image
    """
    assert 0.0 < gain < 2.0
    assert niter > 0
    assert len(scales) > 0
    
    comps = numpy.zeros(dirty.shape)
    
    pmax = psf.max()
    assert pmax > 0.0
    
    # Rescale to unit peak PSF. We undo this at the end of iteration
    psfpeak = argmax(numpy.fabs(psf))
    log.info("msclean: Peak of PSF = %s at %s" % (pmax, psfpeak))
    dmax = dirty.max()
    dpeak = argmax(dirty)
    log.info("msclean: Peak of Dirty = %s at %s" % (dmax, dpeak))
    lpsf = psf / pmax
    ldirty = dirty / pmax
    
    # Create the scale images and form all the various products we need. We
    # use an extra dimension to hold the scale-related images. scalestack is a 3D
    # cube holding the different scale images. convolvestack will take a 2D Image
    # and add a third dimension holding the scale-convolved versions.
    
    scaleshape = [len(scales), ldirty.shape[0], ldirty.shape[1]]
    scalestack = create_scalestack(scaleshape, scales, norm=True)
    
    pscaleshape = [len(scales), lpsf.shape[0], lpsf.shape[1]]
    pscalescaleshape = [len(scales), len(scales), lpsf.shape[0], lpsf.shape[1]]
    pscalestack = create_scalestack(pscaleshape, scales, norm=True)
    
    psf_scalestack = convolve_scalestack(pscalestack, numpy.array(lpsf))
    res_scalestack = convolve_scalestack(scalestack, numpy.array(ldirty))
    
    # Evaluate the coupling matrix between the various scale sizes.
    psf_scalescalestack = numpy.zeros(pscalescaleshape)
    for iscale in numpy.arange(len(scales)):
        psf_scalescalestack[:, iscale, :, :] = convolve_scalestack(pscalestack, psf_scalestack[iscale, :, :])
        psf_scalescalestack[iscale, :, :, :] = psf_scalescalestack[:, iscale, :, :]
    coupling_matrix = numpy.zeros([len(scales), len(scales)])
    for iscale in numpy.arange(len(scales)):
        for iscale1 in numpy.arange(len(scales)):
            coupling_matrix[iscale, iscale1] = numpy.max(psf_scalescalestack[iscale, iscale1, :, :])
    log.info("msclean: Coupling matrix =\n %s" % coupling_matrix)
    
    # The window is scale dependent - we form it by smoothing and thresholding
    # the input window. This prevents components being placed too close to the
    # edge of the Image.
    
    if window is None:
        windowstack = None
    else:
        windowstack = numpy.zeros_like(scalestack)
        windowstack[convolve_scalestack(scalestack, window) > 0.9] = 1.0
    
    log.info("msclean: Max abs in dirty Image = %.6f" % numpy.fabs(res_scalestack[0, :, :]).max())
    absolutethresh = max(thresh, fracthresh * numpy.fabs(res_scalestack[0, :, :]).max())
    log.info("msclean: Start of minor cycle")
    log.info("msclean: This minor cycle will stop at %d iterations or peak < %s" % (niter, absolutethresh))
    
    for i in range(niter):
        # Find peak over all smoothed images
        mx, my, mscale = find_max_abs_stack(res_scalestack, windowstack, coupling_matrix)
        if mx is None or my is None or mscale is None:
            raise RuntimeError("msclean: Error in finding peak")
        
        # Find the values to subtract, accounting for the coupling matrix
        mval = numpy.zeros(len(scales))
        mval[mscale] = res_scalestack[mscale, mx, my] / coupling_matrix[mscale, mscale]
        if i % (niter // 10) == 0:
            log.info("msclean: Minor cycle %d, peak %s at [%d, %d, %d]" % \
                     (i, res_scalestack[:, mx, my], mx, my, mscale))
        if numpy.fabs(mval[mscale]) < absolutethresh:
            log.info("msclean: Absolute value of peak %.6f is below stopping threshold %.6f" \
                     % (numpy.fabs(res_scalestack[mscale, mx, my]), absolutethresh))
            break
        
        # Update the cached residuals and add to the cached model.
        a1o, a2o = overlapIndices(dirty, psf, mx, my)
        if numpy.abs(mval[mscale]) > 0:
            # Cross subtract from other scales
            for iscale in range(len(scales)):
                res_scalestack[iscale, a1o[0]:a1o[1], a1o[2]:a1o[3]] -= \
                    psf_scalescalestack[iscale, mscale, a2o[0]:a2o[1], a2o[2]:a2o[3]] * gain * mval[mscale]
            comps[a1o[0]:a1o[1], a1o[2]:a1o[3]] += \
                pscalestack[mscale, a2o[0]:a2o[1], a2o[2]:a2o[3]] * gain * mval[mscale]
        else:
            break
    log.info("msclean: End of minor cycles")
    return comps, pmax * res_scalestack[0, :, :]


def create_scalestack(scaleshape, scales, norm=True):
    """ Create a cube consisting of the scales

    :param scaleshape: desired shape of stack
    :param scales: scales (in pixels)
    :param norm: Normalise each plane to unity?
    :returns: stack
    """
    assert scaleshape[0] == len(scales)
    
    basis = numpy.zeros(scaleshape)
    nx = scaleshape[1]
    ny = scaleshape[2]
    xcen = int(numpy.ceil(float(nx) / 2.0))
    ycen = int(numpy.ceil(float(ny) / 2.0))
    for iscale in numpy.arange(0, len(scales)):
        halfscale = int(numpy.ceil(scales[iscale] / 2.0))
        if scales[iscale] > 0.0:
            rscale2 = 1.0 / (float(scales[iscale]) / 2.0) ** 2
            x = range(xcen - halfscale - 1, xcen + halfscale + 1)
            fx = numpy.array(x, 'float') - float(xcen)
            # Unroll this since spheroidal_function needs a scalar
            for y in range(ycen - halfscale - 1, ycen + halfscale + 1):
                for x in range(xcen - halfscale - 1, xcen + halfscale + 1):
                    fx = float(x - xcen)
                    fy = float(y - ycen)
                    r2 = rscale2 * (fx * fx + fy * fy)
                    r = numpy.sqrt(r2)
                    basis[iscale, x, y] = spheroidal_function(r) * (1.0 - r ** 2)
            basis[basis < 0.0] = 0.0
            if norm:
                basis[iscale, :, :] /= numpy.sum(basis[iscale, :, :])
        else:
            basis[iscale, xcen, ycen] = 1.0
    return basis


def convolve_scalestack(scalestack, img):
    """Convolve img by the specified scalestack, returning the resulting stack

    :param scalestack: stack containing the scales
    :param img: Image to be convolved
    :returns: stack
    """
    
    convolved = numpy.zeros(scalestack.shape)
    ximg = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(img)))
    
    nscales = scalestack.shape[0]
    for iscale in range(nscales):
        xscale = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(scalestack[iscale, :, :])))
        xmult = ximg * xscale
        convolved[iscale, :, :] = numpy.real(numpy.fft.ifftshift(numpy.fft.ifft2(numpy.fft.ifftshift(xmult))))
    return convolved


def convolve_convolve_scalestack(scalestack, img):
    """Convolve img by the specified scalestack, returning the resulting stack

    :param scalestack: stack containing the scales
    :param img: Image to be convolved
    :returns: Twice convolved image [nscales, nscales, nx, ny]
    """
    
    nscales, nx, ny = scalestack.shape
    convolved_shape = [nscales, nscales, nx, ny]
    convolved = numpy.zeros(convolved_shape)
    ximg = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(img)))

    xscaleshape = [nscales, nx, ny]
    xscale = numpy.zeros(xscaleshape, dtype='complex')
    for s in range(nscales):
        xscale[s] = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(scalestack[s,...])))

    for s in range(nscales):
        for p in range(nscales):
            xmult = ximg * xscale[p] * xscale[s]
            convolved[s, p, ...] = numpy.real(numpy.fft.ifftshift(numpy.fft.ifft2(numpy.fft.ifftshift(xmult))))
    return convolved


def find_max_abs_stack(stack, windowstack, couplingmatrix):
    """Find the location and value of the absolute maximum in this stack
    :param stack: stack to be searched
    :param windowstack: Window for the search
    :param couplingmatrix: Coupling matrix between difference scales
    :returns: x, y, scale

    """
    pabsmax = 0.0
    pscale = None
    px = None
    py = None
    pshape = [stack.shape[1], stack.shape[2]]
    for iscale in range(stack.shape[0]):
        if windowstack is not None:
            resid = stack[iscale, :, :] * windowstack[iscale, :, :]
        else:
            resid = stack[iscale, :, :]
        
        mx, my = numpy.unravel_index(numpy.fabs(resid).argmax(), pshape)
        thisabsmax = stack[iscale, mx, my] / couplingmatrix[iscale, iscale]
        if abs(thisabsmax) > abs(pabsmax):
            px = mx
            py = my
            pscale = iscale
            pabsmax = stack[pscale, px, py] / couplingmatrix[iscale, iscale]
    return px, py, pscale


def spheroidal_function(vnu):
    """ Evaluates the PROLATE SPHEROIDAL WAVEFUNCTION

    m=6, alpha = 1 from Schwab, Indirect Imaging (1984).
    This is one factor in the basis function.
    """
    
    # Code adapted Anna's f90 PROFILE (gridder.f90) code
    # which was adapted from Tim Cornwell's C++ SphFuncVisGridder
    # developed for CONRAD for ASKAP. **This seems to be commented
    # out of the currect ASKAPsoft code... not sure why**
    #
    # Stole this back from Anna!
    n_p = 4
    n_q = 2
    
    p = numpy.zeros((2, 5))
    q = numpy.zeros((2, 3))
    
    p[0, 0] = 8.203343e-2
    p[0, 1] = -3.644705e-1
    p[0, 2] = 6.278660e-1
    p[0, 3] = -5.335581e-1
    p[0, 4] = 2.312756e-1
    p[1, 0] = 4.028559e-3
    p[1, 1] = -3.697768e-2
    p[1, 2] = 1.021332e-1
    p[1, 3] = -1.201436e-1
    p[1, 4] = 6.412774e-2
    
    q[0, 0] = 1.0000000
    q[0, 1] = 8.212018e-1
    q[0, 2] = 2.078043e-1
    q[1, 0] = 1.0000000
    q[1, 1] = 9.599102e-1
    q[1, 2] = 2.918724e-1
    
    value = 0.
    
    if (vnu >= 0.) and (vnu < 0.75):
        part = 0
        nuend = 0.75
    elif (vnu >= 0.75) and (vnu <= 1.):
        part = 1
        nuend = 1.0
    else:
        value = 0.
        # nasty fortran-esque exit statement:
        return value
    
    top = p[part, 0]
    bot = q[part, 0]
    delnusq = vnu ** 2 - nuend ** 2
    
    for k in range(1, n_p + 1):
        factor = delnusq ** k
        top += p[part, k] * factor
    
    for k in range(1, n_q + 1):
        factor = delnusq ** k
        bot += q[part, k] * factor
    
    if bot != 0.:
        value = top / bot
    else:
        value = 0.
    
    if value < 0.:
        value = 0.
    
    return value
