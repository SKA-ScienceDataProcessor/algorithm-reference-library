""" Image Deconvolution functions

"""

__all__ = ['hogbom', 'hogbom_complex', 'msclean', 'msmfsclean', 'spheroidal_function']

import numpy
import logging
import time

log = logging.getLogger(__name__)


def hogbom(dirty, psf, window, gain, thresh, niter, fracthresh, prefix=''):
    """ Clean the point spread function from a dirty image

    See Hogbom CLEAN (1974A&AS...15..417H)

    This version operates on numpy arrays.

    :param fracthresh:
    :param prefix:
    :param dirty: The dirty Image, i.e., the Image to be deconvolved
    :param psf: The point spread-function
    :param window: Regions where clean components are allowed. If True, entire dirty Image is allowed
    :param gain: The "loop gain", i.e., the fraction of the brightest pixel that is removed in each iteration
    :param thresh: Cleaning stops when the maximum of the absolute deviation of the residual is less than this value
    :param niter: Maximum number of components to make if the threshold `thresh` is not hit
    :return: clean component Image, residual Image
    """

    starttime = time.time()
    assert 0.0 < gain < 2.0
    assert niter > 0
    
    log.info("hogbom %s Max abs in dirty image = %.6f Jy/beam" % (prefix, numpy.max(numpy.abs(dirty))))
    absolutethresh = max(thresh, fracthresh * numpy.fabs(dirty).max())
    log.info("hogbom %s Start of minor cycle" % prefix)
    log.info("hogbom %s This minor cycle will stop at %d iterations or peak < %.6f (Jy/beam)" %
             (prefix, niter, absolutethresh))

    comps = numpy.zeros(dirty.shape)
    res = numpy.array(dirty)
    pmax = psf.max()
    assert pmax > 0.0
    log.info('hogbom %s: Timing for setup: %.3f (s) for dirty shape %s, PSF shape %s' %
             (prefix, time.time() - starttime, str(dirty.shape), str(psf.shape)))
    starttime = time.time()
    aiter = 0
    for i in range(niter):
        aiter = i + 1
        if window is not None:
            mx, my = numpy.unravel_index((numpy.fabs(res * window)).argmax(), dirty.shape)
        else:
            mx, my = numpy.unravel_index((numpy.fabs(res)).argmax(), dirty.shape)
        mval = res[mx, my] * gain / pmax
        comps[mx, my] += mval
        a1o, a2o = overlapIndices(dirty, psf, mx, my)
        if niter < 10 or i % (niter // 10) == 0:
            log.info("hogbom %s Minor cycle %d, peak %s at [%d, %d]" % (prefix, i, res[mx, my], mx, my))
        res[a1o[0]:a1o[1], a1o[2]:a1o[3]] -= psf[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval
        if numpy.abs(res[mx, my]) < 0.9 * absolutethresh:
            log.info("hogbom %s Stopped at iteration %d, peak %s at [%d, %d]" % (prefix, i, res[mx, my], mx, my))
            break
    log.info("hogbom %s End of minor cycle" % prefix)
    
    dtime = time.time() - starttime
    log.info('%s Timing for clean: %.3f (s) for dirty %s, PSF %s , %d iterations, time per clean %.3f (ms)' %
             (prefix, dtime, str(dirty.shape), str(psf.shape), aiter, 1000.0 * dtime / aiter))

    return comps, res


def hogbom_complex(dirty_q, dirty_u, psf_q, psf_u, window, gain, thresh, niter, fracthresh):
    """Clean the point spread function from a dirty Q+iU image

    This uses the complex Hogbom CLEAN for polarised data (2016MNRAS.462.3483P)

    The starting-point for the code was the standard Hogbom clean algorithm available in ARL.

    Args:
    dirty_q (numpy array): The dirty Q Image, i.e., the Q Image to be deconvolved.
    dirty_u (numpy array): The dirty U Image, i.e., the U Image to be deconvolved.
    psf_q (numpy array): The point spread-function in Stokes Q.
    psf_u (numpy array): The point spread-function in Stokes U.
    window (float): Regions where clean components are allowed. If True, entire dirty Image is allowed.
    gain (float): The "loop gain", i.e., the fraction of the brightest pixel that is removed in each iteration.
    thresh (float): Cleaning stops when the maximum of the absolute deviation of the residual is less than this value.
    niter (int): Maximum number of components to make if the threshold `thresh` is not hit.
    fracthresh (float): The predefined fractional threshold at which to stop cleaning.

    Returns:
    comps.real: real clean component image.
    comps.imag: imaginary clean component image.
    res.real: real residual image.
    res.imag: imaginary residual image.
    """
    
    assert 0.0 < gain < 2.0
    assert niter > 0
    
    # Form complex Q+iU from the polarisation data:
    dirty_complex = dirty_q + 1j * dirty_u

    log.info("hogbom_mod: Max abs in dirty image = %.6f" % numpy.max(numpy.abs(dirty_complex)))
    absolutethresh = max(thresh, fracthresh * numpy.absolute(dirty_complex).max())
    log.info("hogbom_mod: Start of minor cycle")
    log.info("hogbom_mod: This minor cycle will stop at %d iterations or peak < %s" % (niter, absolutethresh))
    
    comps = numpy.zeros(dirty_complex.shape, dtype='complex128')
    res = numpy.array(dirty_complex)
    
    assert numpy.all(psf_q == psf_u)
    
    pmax = psf_q.max()
    assert pmax > 0.0
    log.info("hogbom: Max abs in dirty Image = %.6f" % numpy.absolute(res).max())
    for i in range(niter):
        if window is not None:
            mx, my = numpy.unravel_index((numpy.absolute(res * window)).argmax(), dirty_complex.shape)
        else:
            mx, my = numpy.unravel_index((numpy.absolute(res)).argmax(), dirty_complex.shape)
        mval = res[mx, my] * gain / pmax
        comps[mx, my] += mval
        a1o, a2o = overlapIndices(dirty_complex, psf_q, mx, my)
        if niter < 10 or i % (niter // 10) == 0:
            log.info("hogbom: Minor cycle %d, peak %s at [%d, %d]" % (i, res[mx, my], mx, my))
        res[a1o[0]:a1o[1], a1o[2]:a1o[3]] -= psf_q[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval
        if numpy.abs(res[mx, my]) < absolutethresh:
            log.info("hogbom: Stopped at iteration %d, peak %s at [%d, %d]" % (i, res[mx, my], mx, my))
            break
    log.info("hogbom: End of minor cycle")
    return comps.real, comps.imag, res.real, res.imag


def overlapIndices(res, psf, peakx, peaky):
    """ Find the indices where two arrays overlap

    :param res:
    :param psf:
    :param peakx: peak in x
    :param peaky: peak in y
    :return: (limits in a1, limits in a2)
    """
    nx, ny = res.shape[0], res.shape[1]
    psfwidthx, psfwidthy = psf.shape[0] // 2, psf.shape[1] // 2
    psfpeakx, psfpeaky = psf.shape[0] // 2, psf.shape[1] // 2
    # Step 1 line up the coordinate ignoring limits
    res_lower = (max(0, peakx - psfwidthx), max(0, peaky - psfwidthy))
    res_upper = (min(nx, peakx + psfwidthx), min(peaky + psfwidthy, ny))
    psf_lower = (max(0, psfpeakx + (res_lower[0] - peakx)), max(0, psfpeaky + (res_lower[1] - peaky)))
    psf_upper = (
        min(psf.shape[0], psfpeakx + (res_upper[0] - peakx)), min(psfpeaky + (res_upper[1] - peaky), psf.shape[1]))

    return (res_lower[0], res_upper[0], res_lower[1], res_upper[1]), \
           (psf_lower[0], psf_upper[0], psf_lower[1], psf_upper[1])


def argmax(a):
    """ Return unravelled index of the maximum

    param: a: array to be searched
    """
    return numpy.unravel_index(a.argmax(), a.shape)


def msclean(dirty, psf, window, gain, thresh, niter, scales, fracthresh, prefix=''):
    """ Perform multiscale clean

    Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc, 2008 vol. 2 pp. 793-801)

    This version operates on numpy arrays.

    :param prefix:
    :param fracthresh:
    :param dirty: The dirty image, i.e., the image to be deconvolved
    :param psf: The point spread-function
    :param window: Regions where clean components are allowed. If True, all of the dirty image is allowed
    :param gain: The "loop gain", i.e., the fraction of the brightest pixel that is removed in each iteration
    :param thresh: Cleaning stops when the maximum of the absolute deviation of the residual is less than this value
    :param niter: Maximum number of components to make if the threshold "thresh" is not hit
    :param scales: Scales (in pixels width) to be used
    :return: clean component image, residual image
    """
    
    starttime = time.time()
    assert 0.0 < gain < 2.0
    assert niter > 0
    assert len(scales) > 0

    comps = numpy.zeros(dirty.shape)

    pmax = psf.max()
    assert pmax > 0.0

    # Rescale to unit peak PSF. We undo this at the end of iteration
    psfpeak = argmax(numpy.fabs(psf))
    log.info("msclean %s: Peak of PSF = %s at %s" % (prefix, pmax, psfpeak))
    dmax = dirty.max()
    dpeak = argmax(dirty)
    log.info("msclean %s: Peak of Dirty = %.6f Jy/beam at %s " % (prefix, dmax, dpeak))
    lpsf = psf / pmax
    ldirty = dirty / pmax

    # Create the scale images and form all the various products we need. We
    # use an extra dimension to hold the scale-related images. scalestack is a 3D
    # cube holding the different scale images. convolvestack will take a 2D Image
    # and add a third dimension holding the scale-convolved versions.

    scaleshape = [len(scales), ldirty.shape[0], ldirty.shape[1]]
    scalestack = create_scalestack(scaleshape, scales, norm=True)

    pscaleshape = [len(scales), lpsf.shape[0], lpsf.shape[1]]
    pscalestack = create_scalestack(pscaleshape, scales, norm=True)

    res_scalestack = convolve_scalestack(scalestack, numpy.array(ldirty))
    psf_scalescalestack = convolve_convolve_scalestack(pscalestack, numpy.array(lpsf))

    # Evaluate the coupling matrix between the various scale sizes.
    coupling_matrix = numpy.zeros([len(scales), len(scales)])
    for iscale in numpy.arange(len(scales)):
        for iscale1 in numpy.arange(len(scales)):
            coupling_matrix[iscale, iscale1] = numpy.max(psf_scalescalestack[iscale, iscale1, :, :])
    log.info("msclean %s: Coupling matrix =\n %s" % (prefix, coupling_matrix))

    # The window is scale dependent - we form it by smoothing and thresholding
    # the input window. This prevents components being placed too close to the
    # edge of the Image.

    if window is None:
        windowstack = None
    else:
        windowstack = numpy.zeros_like(scalestack)
        windowstack[convolve_scalestack(scalestack, window) > 0.9] = 1.0

    if windowstack is not None:
        assert numpy.sum(windowstack) > 0

    log.info("msclean %s: Max abs in dirty Image = %.6f Jy/beam" % (prefix, numpy.fabs(res_scalestack[0, :, :]).max()))
    absolutethresh = max(thresh, fracthresh * numpy.fabs(res_scalestack[0, :, :]).max())
    log.info("msclean %s: Start of minor cycle" % prefix)
    log.info("msclean %s: This minor cycle will stop at %d iterations or peak < %.6f (Jy/beam)" %
             (prefix, niter, absolutethresh))

    log.info('msclean %s: Timing for setup: %.3f (s) for dirty shape %s, PSF shape %s , scales %s' %
             (prefix, time.time() - starttime, str(dirty.shape), str(psf.shape), str(scales)))
    starttime = time.time()
    aiter = 0
    for i in range(niter):
        aiter = i + 1
        # Find peak over all smoothed images
        mx, my, mscale = find_max_abs_stack(res_scalestack, windowstack, coupling_matrix)
        # Find the values to subtract, accounting for the coupling matrix
        mval = res_scalestack[mscale, mx, my] / coupling_matrix[mscale, mscale]
        if niter < 10 or i % (niter // 10) == 0:
            log.info("msclean %s: Minor cycle %d, peak %s at [%d, %d, %d]" %
                     (prefix, i, res_scalestack[:, mx, my], mx, my, mscale))
        if numpy.fabs(res_scalestack[mscale, mx, my]) < 0.9 * absolutethresh:
            log.info("msclean %s: At iteration %d, absolute value of peak %.6f is below stopping threshold %.6f"
                     % (prefix, i, numpy.fabs(res_scalestack[mscale, mx, my]), absolutethresh))
            break

        # Update the cached residuals and add to the cached model.
        lhs, rhs = overlapIndices(dirty, psf, mx, my)
        if numpy.abs(mval) > 0:
            # Cross subtract from other scales
            for iscale in range(len(scales)):
                res_scalestack[iscale, lhs[0]:lhs[1], lhs[2]:lhs[3]] -= \
                    psf_scalescalestack[iscale, mscale, rhs[0]:rhs[1], rhs[2]:rhs[3]] * gain * mval
            comps[lhs[0]:lhs[1], lhs[2]:lhs[3]] += \
                pscalestack[mscale, rhs[0]:rhs[1], rhs[2]:rhs[3]] * gain * mval
        else:
            break
            
    log.info("msclean %s: End of minor cycle" % prefix)
    
    dtime = time.time() - starttime
    log.info('msclean %s: Timing for clean: %.3f (s) for dirty shape %s, PSF shape %s , scales %s, %d iterations, '
             'time per clean %.3f (ms)' %
             (prefix, dtime, str(dirty.shape), str(psf.shape), str(scales), aiter, 1000.0 * dtime / aiter))

    return comps, pmax * res_scalestack[0, :, :]


def create_scalestack(scaleshape, scales, norm=True):
    """ Create a cube consisting of the scales

    :param scaleshape: desired shape of stack
    :param scales: scales (in pixels)
    :param norm: Normalise each plane to unity?
    :return: stack
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
    :return: stack
    """

    convolved = numpy.zeros(scalestack.shape)
    ximg = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(img)))

    nscales = scalestack.shape[0]
    for iscale in range(nscales):
        xscale = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(scalestack[iscale, :, :])))
        xmult = ximg * numpy.conjugate(xscale)
        convolved[iscale, :, :] = numpy.real(numpy.fft.ifftshift(numpy.fft.ifft2(numpy.fft.ifftshift(xmult))))
    return convolved


def convolve_convolve_scalestack(scalestack, img):
    """Convolve img by the specified scalestack, returning the resulting stack

    :param scalestack: stack containing the scales
    :param img: Image to be convolved
    :return: Twice convolved image [nscales, nscales, nx, ny]
    """

    nscales, nx, ny = scalestack.shape
    convolved_shape = [nscales, nscales, nx, ny]
    convolved = numpy.zeros(convolved_shape)
    ximg = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(img)))

    xscaleshape = [nscales, nx, ny]
    xscale = numpy.zeros(xscaleshape, dtype='complex')
    for s in range(nscales):
        xscale[s] = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(scalestack[s, ...])))

    for s in range(nscales):
        for p in range(nscales):
            xmult = ximg * xscale[p] * numpy.conjugate(xscale[s])
            convolved[s, p, ...] = numpy.real(numpy.fft.ifftshift(numpy.fft.ifft2(numpy.fft.ifftshift(xmult))))
    return convolved


def find_max_abs_stack(stack, windowstack, couplingmatrix):
    """Find the location and value of the absolute maximum in this stack
    :param stack: stack to be searched
    :param windowstack: Window for the search
    :param couplingmatrix: Coupling matrix between difference scales
    :return: x, y, scale

    """
    pabsmax = 0.0
    pscale = 0
    px = 0
    py = 0
    nscales = stack.shape[0]
    assert nscales > 0
    pshape = [stack.shape[1], stack.shape[2]]
    for iscale in range(nscales):
        if windowstack is not None:
            resid = stack[iscale, :, :] * windowstack[iscale, :, :] / couplingmatrix[iscale, iscale]
        else:
            resid = stack[iscale, :, :] / couplingmatrix[iscale, iscale]

        # Find the peak in the scaled residual image
        mx, my = numpy.unravel_index(numpy.abs(resid).argmax(), pshape)

        # Is this the peak over all scales?
        thisabsmax = numpy.abs(resid[mx, my])
        if thisabsmax > pabsmax:
            px = mx
            py = my
            pscale = iscale
            pabsmax = thisabsmax

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


def msmfsclean(dirty, psf, window, gain, thresh, niter, scales, fracthresh, findpeak='ARL', prefix=''):
    """ Perform image plane multiscale multi frequency clean

    This algorithm is documented as Algorithm 1 in: U. Rau and T. J. Cornwell, “A multi-scale multi-frequency
    deconvolution algorithm for synthesis imaging in radio interferometry,” A&A 532, A71 (2011). Note that
    this is only the image plane parts.

    Specific code is linked to specific lines in that algorithm description.

    This version operates on numpy arrays that have been converted to moments on the last axis.

    :param fracthresh:
    :param dirty: The dirty image, i.e., the image to be deconvolved
    :param psf: The point spread-function
    :param window: Regions where clean components are allowed. If True, all of the dirty image is allowed
    :param gain: The "loop gain", i.e., the fraction of the brightest pixel that is removed in each iteration
    :param thresh: Cleaning stops when the maximum of the absolute deviation of the residual is less than this value
    :param niter: Maximum number of components to make if the threshold "thresh" is not hit
    :param scales: Scales (in pixels width) to be used
    :param fracthresh: Fractional stopping threshold
    :param findpeak: Method of finding peak in mfsclean: 'Algorithm1'|'CASA'|'ARL', Default is ARL.
    :param prefix: Prefix to log messages to provide context
    :return: clean component image, residual image
    """
    
    starttime = time.time()
    assert 0.0 < gain < 2.0
    assert niter > 0
    assert len(scales) > 0

    m_model = numpy.zeros(dirty.shape)

    nscales = len(scales)

    pmax = psf.max()
    assert pmax > 0.0

    psfpeak = argmax(numpy.fabs(psf))
    log.info("mmclean %s: Peak of PSF = %s at %s" % (prefix, pmax, psfpeak))
    dmax = dirty.max()
    dpeak = argmax(dirty)
    log.info("mmclean %s: Peak of Dirty = %.6f Jy/beam at %s " % (prefix, dmax, dpeak))
    lpsf = psf / pmax
    ldirty = dirty / pmax

    nmoment, ny, nx = dirty.shape
    if nmoment > 1:
        assert psf.shape[0] == 2 * nmoment

    # Create the "scale basis functions" in Algorithm 1
    scaleshape = [nscales, ldirty.shape[1], ldirty.shape[2]]
    scalestack = create_scalestack(scaleshape, scales, norm=True)

    pscaleshape = [nscales, lpsf.shape[1], lpsf.shape[2]]
    pscalestack = create_scalestack(pscaleshape, scales, norm=True)

    # Calculate scale convolutions of moment residuals
    smresidual = calculate_scale_moment_residual(ldirty, scalestack)

    # Calculate scale scale moment moment psf, Hessian, and inverse of Hessian
    # scale scale moment moment psf is needed for update of scale-moment residuals
    # Hessian is needed in calculation of optimum for any iteration
    # Inverse Hessian is needed to calculate principal solution in moment-space
    ssmmpsf = calculate_scale_scale_moment_moment_psf(lpsf, pscalestack)
    hsmmpsf, ihsmmpsf = calculate_scale_inverse_moment_moment_hessian(ssmmpsf)

    for scale in range(nscales):
        log.debug("mmclean %s: Moment-moment coupling matrix[scale %d] =\n %s" % (prefix, scale, hsmmpsf[scale]))

    # The window is scale dependent - we form it by smoothing and thresholding
    # the input window. This prevents components being placed too close to the
    # edge of the Image.

    if window is None:
        windowstack = None
    else:
        windowstack = numpy.zeros_like(scalestack)
        windowstack[convolve_scalestack(scalestack, window) > 0.9] = 1.0

    log.info("mmclean %s: Max abs in dirty Image = %.6f Jy/beam" % (prefix, numpy.fabs(smresidual[0, 0, :, :]).max()))
    absolutethresh = max(thresh, fracthresh * numpy.fabs(smresidual[0, 0, :, :]).max())
    log.info("mmclean %s: Start of minor cycle" % prefix)
    log.info("mmclean %s: This minor cycle will stop at %d iterations or peak < %.6f (Jy/beam)" %
             (prefix, niter, absolutethresh))

    # Start iterations
    scale_counts = numpy.zeros(nscales, dtype='int')
    scale_flux = numpy.zeros(nscales)

    aiter = 0
    log.info('mmclean %s: Timing for setup: %.3f (s) for dirty shape %s, PSF shape %s , scales %s, %d moments' %
             (prefix, time.time() - starttime, str(dirty.shape), str(psf.shape), str(scales), nmoment))
    starttime = time.time()
    for i in range(niter):
        aiter = i + 1

        # Find the optimum scale and location.
        mscale, mx, my, mval = find_global_optimum(hsmmpsf, ihsmmpsf, smresidual, windowstack, findpeak)
        scale_counts[mscale] += 1
        scale_flux[mscale] += mval[0]

        # Report on progress
        if niter < 10 or i % (niter // 10) == 0:
            log.info("mmclean %s: Minor cycle %d, peak %s at [%d, %d, %d]" % (prefix, i, mval, mx, my, mscale))

        # Are we ready to stop yet?
        peak = numpy.max(numpy.fabs(mval))
        if peak < absolutethresh:
            log.info("mmclean %s: At iteration %d, absolute value of peak %.6f is below stopping threshold %.6f"
                     % (prefix, i, peak, absolutethresh))
            break

        # Calculate indices needed for lhs and rhs of updates to model and residual
        lhs, rhs = overlapIndices(ldirty[0, ...], psf[0, ...], mx, my)

        # Update model and residual image
        m_model = update_moment_model(m_model, pscalestack, lhs, rhs, gain, mscale, mval)
        smresidual = update_scale_moment_residual(smresidual, ssmmpsf, lhs, rhs, gain, mscale, mval)

    log.info("mmclean %s: End of minor cycles" % prefix)

    log.info("mmclean %s: Scale counts %s" % (prefix, scale_counts))
    log.info("mmclean %s: Scale flux %s" % (prefix, scale_flux))

    dtime = time.time() - starttime
    log.info('mmclean %s: Timing for clean: %.3f (s) for dirty shape %s, PSF shape %s , scales %s, %d moments, '
             '%d iterations, time per clean %.3f (ms)' %
             (prefix, dtime, str(dirty.shape), str(psf.shape), str(scales), nmoment, aiter, 1000.0 * dtime / aiter))

    return m_model, pmax * smresidual[0, :, :, :]


def find_global_optimum(hsmmpsf, ihsmmpsf, smresidual, windowstack, findpeak):
    """Find the optimum peak using one of a number of algorithms

    """
    if findpeak == 'Algorithm1':
        # Calculate the principal solution in moment-moment axes. This decouples the moments
        smpsol = calculate_scale_moment_principal_solution(smresidual, ihsmmpsf)
        # Now find the location and scale
        mx, my, mscale = find_optimum_scale_zero_moment(smpsol, windowstack)
        mval = smpsol[mscale, :, mx, my]
    elif findpeak == 'CASA':
        # CASA 4.7 version
        smpsol = calculate_scale_moment_principal_solution(smresidual, ihsmmpsf)
        #        smpsol = calculate_scale_moment_approximate_principal_solution(smresidual, hsmmpsf)
        nscales, nmoment, nx, ny = smpsol.shape  # pylint: disable=no-member
        dchisq = numpy.zeros([nscales, 1, nx, ny])
        for scale in range(nscales):
            for moment1 in range(nmoment):
                dchisq[scale, 0, ...] += 2.0 * smpsol[scale, moment1, ...] * smresidual[scale, moment1, ...]
                for moment2 in range(nmoment):
                    dchisq[scale, 0, ...] -= hsmmpsf[scale, moment1, moment2] * \
                        smpsol[scale, moment1, ...] * smpsol[scale, moment2, ...]

        mx, my, mscale = find_optimum_scale_zero_moment(dchisq, windowstack)
        mval = smpsol[mscale, :, mx, my]

    else:
        smpsol = calculate_scale_moment_principal_solution(smresidual, ihsmmpsf)
        mx, my, mscale = find_optimum_scale_zero_moment(smpsol * smresidual, windowstack)

        mval = smpsol[mscale, :, mx, my]

    return mscale, mx, my, mval


def update_scale_moment_residual(smresidual, ssmmpsf, lhs, rhs, gain, mscale, mval):
    """ Update residual by subtracting the effect of model update for each moment

    """
    # Lines 30 - 32 of Algorithm 1.
    nscales, nmoment, _, _ = smresidual.shape
    smresidual[:, :, lhs[0]:lhs[1], lhs[2]:lhs[3]] -= \
        gain * numpy.einsum("stqxy,q->stxy", ssmmpsf[mscale, :, :, :, rhs[0]:rhs[1], rhs[2]:rhs[3]], mval)

    return smresidual


def update_moment_model(m_model, scalestack, lhs, rhs, gain, mscale, mval):
    """Update model with an appropriately scaled and centered blob for each moment

    """
    # Lines 28 - 33 of Algorithm 1
    nmoment, _, _ = m_model.shape
    for t in range(nmoment):
        # Line 29 of Algorithm 1. Note that the convolution is implemented here as an
        # appropriate shift.
        m_model[t, lhs[0]:lhs[1], lhs[2]:lhs[3]] += \
            scalestack[mscale, rhs[0]:rhs[1], rhs[2]:rhs[3]] * gain * mval[t]

    return m_model


def calculate_scale_moment_residual(residual, scalestack):
    """ Calculate scale-dependent moment residuals

    Part of the initialisation for Algorithm 1: lines 12 - 17

    :param scalestack:
    :param residual: residual [nmoment, nx, ny]
    :return: scale-dependent moment residual [nscales, nmoment, nx, ny]
    """
    nmoment, nx, ny = residual.shape
    nscales = scalestack.shape[0]

    # Lines 12 - 17 from Algorithm 1
    scale_moment_residual = numpy.zeros([nscales, nmoment, nx, ny])
    for t in range(nmoment):
        scale_moment_residual[:, t, ...] = convolve_scalestack(scalestack, residual[t, ...])
    return scale_moment_residual


def calculate_scale_scale_moment_moment_psf(psf, scalestack):
    """ Calculate scale-dependent moment psfs

    Part of the initialisation for Algorithm 1

    :param scalestack:
    :param psf: psf
    :return: scale-dependent moment psf [nscales, nscales, nmoment, nmoment, nx, ny]
    """
    nmoment2, nx, ny = psf.shape
    nmoment = max(nmoment2 // 2, 1)
    nscales = scalestack.shape[0]

    # Lines 3 - 5 from Algorithm 1
    scale_scale_moment_moment_psf = numpy.zeros([nscales, nscales, nmoment, nmoment, nx, ny])
    for t in range(nmoment):
        for q in range(nmoment):
            scale_scale_moment_moment_psf[:, :, t, q] = convolve_convolve_scalestack(scalestack, psf[t + q])
    return scale_scale_moment_moment_psf


def calculate_scale_inverse_moment_moment_hessian(scale_scale_moment_moment_psf):
    """Calculate inverse_scale dependent moment moment hessian

    Part of the initialisation for Algorithm 1. Lines 7 - 9

    :param scale_scale_moment_moment_psf: scale_moment_psf [nscales, nscales, nmoment, nmoment]
    :return: scale-dependent moment-moment inverse hessian
    """
    nscales, _, nmoment, _, nx, ny = scale_scale_moment_moment_psf.shape
    hessian_shape = [nscales, nmoment, nmoment]

    scale_moment_moment_hessian = numpy.zeros(hessian_shape)
    scale_inverse_moment_moment_hessian = numpy.zeros(hessian_shape)
    for s in range(nscales):
        scale_moment_moment_hessian[s, ...] = scale_scale_moment_moment_psf[s, s, ..., nx // 2, ny // 2]
        scale_inverse_moment_moment_hessian[s] = numpy.linalg.inv(scale_moment_moment_hessian[s])
    return scale_moment_moment_hessian, scale_inverse_moment_moment_hessian


def calculate_scale_moment_principal_solution(smresidual, ihsmmpsf):
    """ Calculate the principal solution in moment space for each scale

    Lines 20 - 26

    :param smresidual: scale-dependent moment residual [nscales, nmoment, nx, ny]
    :param ihsmmpsf: Inverse of scale dependent moment moment Hessian
    :return: Decoupled residual images [nscales, nmoment, nx, ny]
    """
    # ihsmmpsf: nscales, nmoment, nmoment
    # smresidual: nscales, nmoment, nx, ny
    smpsol = numpy.einsum("smn,smxy->snxy", ihsmmpsf, smresidual)

    return smpsol


def find_optimum_scale_zero_moment(smpsol, windowstack):
    """Find the optimum scale for moment zero

    Line 27 of Algorithm 1

    :param windowstack:
    :param smpsol: Decoupled residual images for each scale and moment
    :return: x, y, optimum scale for peak
    """
    nscales, nmoment, nx, ny = smpsol.shape
    sscale = 0
    sx = 0
    sy = 0
    optimum = 0.0

    for scale in range(nscales):

        if windowstack is not None:
            resid = smpsol[scale, 0, :, :] * windowstack[scale, :, :]
        else:
            resid = smpsol[scale, 0, :, :]

        this_max = numpy.max(numpy.abs(resid))
        if this_max > optimum:
            optimum = this_max
            sscale = scale
            sx, sy = argmax(smpsol[scale, 0, ...])

    return sx, sy, sscale
