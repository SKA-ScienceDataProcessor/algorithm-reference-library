# Tim Cornwell <realtimcornwell@gmail.com>
""" Image Deconvolution functions

"""
import logging

import numpy

from arl.image.hogbom import argmax, overlapIndices
from arl.image.msclean import convolve_scalestack, convolve_convolve_scalestack, create_scalestack

log = logging.getLogger(__name__)


def msmfsclean(dirty, psf, window, gain, thresh, niter, scales, fracthresh, findpeak='CASA'):
    """ Perform image plane multiscale multi frequency clean
    
    This algorithm is documented as Algorithm 1 in: U. Rau and T. J. Cornwell, “A multi-scale multi-frequency
    deconvolution algorithm for synthesis imaging in radio interferometry,” A&A 532, A71 (2011). Note that
    this is only the image plane parts.

    Specific code is linked to specific lines in that algorithm description.
    
    This version operates on numpy arrays that have been converted to moments on the last axis.

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
    :param ntaylor: Number of Taylor terms
    :returns: clean component image, residual image
    """
    assert 0.0 < gain < 2.0
    assert niter > 0
    assert len(scales) > 0
    
    m_model = numpy.zeros(dirty.shape)
    
    nscales = len(scales)
    
    pmax = psf.max()
    assert pmax > 0.0
    
    psfpeak = argmax(numpy.fabs(psf))
    log.info("msmfsclean: Peak of PSF = %s at %s" % (pmax, psfpeak))
    dmax = dirty.max()
    dpeak = argmax(dirty)
    log.info("msmfsclean: Peak of Dirty = %s at %s" % (dmax, dpeak))
    lpsf = psf / pmax
    ldirty = dirty / pmax
    
    nmoments, ny, nx = dirty.shape
    assert psf.shape[0] == 2 * nmoments
    
    # Create the "scale basis functions" in Algorithm 1
    scaleshape = [nscales, ldirty.shape[1], ldirty.shape[2]]
    scalestack = create_scalestack(scaleshape, scales, norm=True)

    # Calculate scale convolutions of moment residuals
    smresidual = calculate_scale_moment_residual(ldirty, scalestack)

    # Calculate scale scale moment moment psf, Hessian, and inverse of Hessian
    # scale scale moment moment psf is needed for update of scale-moment residuals
    # Hessian is needed in calculation of optimum for any iteration
    # Inverse Hessian is needed to calculate principal soluation in moment-space
    ssmmpsf = calculate_scale_scale_moment_moment_psf(lpsf, scalestack)
    hsmmpsf, ihsmmpsf = calculate_scale_inverse_moment_moment_hessian(ssmmpsf)
    
    for scale in range(nscales):
        log.info("msmfsclean: Moment-moment coupling matrix[scale %d] =\n %s" % (scale, hsmmpsf[scale]))
    
    # The window is scale dependent - we form it by smoothing and thresholding
    # the input window. This prevents components being placed too close to the
    # edge of the Image.
    
    if window is None:
        windowstack = None
    else:
        windowstack = numpy.zeros_like(scalestack)
        windowstack[convolve_scalestack(scalestack, window) > 0.9] = 1.0
    
    log.info("msmfsclean: Max abs in dirty Image = %.6f" % numpy.fabs(smresidual[0, 0, :, :]).max())
    absolutethresh = max(thresh, fracthresh * numpy.fabs(smresidual[0, 0, :, :]).max())
    log.info("msmfsclean: Start of minor cycle")
    log.info("msmfsclean: This minor cycle will stop at %d iterations or peak < %s" % (niter, absolutethresh))
    
    # Start iterations
    scale_counts = numpy.zeros(nscales, dtype='int')
    scale_flux = numpy.zeros(nscales)

    for i in range(niter):
        
        # Find the optimum scale and location.
        mscale, mx, my, mval = find_global_optimum(hsmmpsf, ihsmmpsf, smresidual, windowstack, findpeak)
        scale_counts[mscale] += 1
        scale_flux[mscale] += mval[0]
        
        # Report on progress
        raw_mval = smresidual[mscale, :, mx, my]
        if i % (niter // 10) == 0:
            log.info("msmfsclean: Minor cycle %d, peak %s at [%d, %d, %d]" % (i, mval, mx, my, mscale))
        
        # Are we ready to stop yet?
        peak = numpy.max(numpy.fabs(mval))
        if  peak < absolutethresh:
            log.info("msmfsclean: Absolute value of peak %.6f is below stopping threshold %.6f" \
                     % (peak, absolutethresh))
            break
        
        # Calculate indices needed for lhs and rhs of updates to model and residual
        lhs, rhs = overlapIndices(ldirty[0, ...], psf[0, ...], mx, my)
        
        # Update model and residual image
        m_model = update_moment_model(m_model, scalestack, lhs, rhs, gain, mscale, mval)
        smresidual = update_scale_moment_residual(smresidual, ssmmpsf, lhs, rhs, gain, mscale, mval)
    
    log.info("msmfsclean: End of minor cycles")

    log.info("msmfsclean: Scale counts %s" % (scale_counts))
    log.info("msmfsclean: Scale flux %s" % (scale_flux))

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
    elif findpeak == 'ASKAP':
        # Calculate the approximate principal solution in moment-moment axes. This decouples the moments and
        # addresses the scaling in scale.
        smpsol = calculate_scale_moment_approximate_principal_solution(smresidual, hsmmpsf)
        
        # Now find the location and scale
        mx, my, mscale = find_optimum_scale_zero_moment(smpsol, windowstack)
        mval = smpsol[mscale, :, mx, my]
    elif findpeak == 'CASA' :
        # CASA 4.7 version
        smpsol = calculate_scale_moment_principal_solution(smresidual, ihsmmpsf)
        #        smpsol = calculate_scale_moment_approximate_principal_solution(smresidual, hsmmpsf)
        nscales, nmoments, nx, ny = smpsol.shape
        dchisq = numpy.zeros([nscales, 1, nx, ny])
        for scale in range(nscales):
            for moment1 in range(nmoments):
                dchisq[scale, 0, ...] += 2.0 * smpsol[scale, moment1, ...] * smresidual[scale, moment1, ...]
                for moment2 in range(nmoments):
                    dchisq[scale, 0, ...] -= hsmmpsf[scale, moment1, moment2] * \
                                             smpsol[scale, moment1, ...] * smpsol[scale, moment2, ...]

        mx, my, mscale = find_optimum_scale_zero_moment(dchisq, windowstack)
        mval = smpsol[mscale, :, mx, my]

    else:
        smpsol = calculate_scale_moment_principal_solution(smresidual, ihsmmpsf)
        mx, my, mscale = find_optimum_scale_zero_moment(smpsol*smresidual, windowstack)

        mval = smpsol[mscale, :, mx, my]

    return mscale, mx, my, mval


def update_scale_moment_residual(smresidual, ssmmpsf, lhs, rhs, gain, mscale, mval):
    """ Update residual by subtracting the effect of model update for each moment
    
    """
    # Lines 30 - 32 of Algorithm 1.
    nscales, nmoments, _, _ = smresidual.shape
    smresidual[:, :, lhs[0]:lhs[1], lhs[2]:lhs[3]] -= \
        gain * numpy.einsum("stqxy,q->stxy", ssmmpsf[mscale,:,:,:,rhs[0]:rhs[1], rhs[2]:rhs[3]], mval)
    
    return smresidual


def update_moment_model(m_model, scalestack, lhs, rhs, gain, mscale, mval):
    """Update model with an appropriately scaled and centered blob for each moment
    
    """
    # Lines 28 - 33 of Algorithm 1
    nmoments, _, _ = m_model.shape
    for t in range(nmoments):
        # Line 29 of Algorithm 1. Note that the convolution is implemented here as an
        # appropriate shift.
        m_model[t, lhs[0]:lhs[1], lhs[2]:lhs[3]] += \
            scalestack[mscale, rhs[0]:rhs[1], rhs[2]:rhs[3]] * gain * mval[t]
    
    return m_model


def calculate_scale_moment_psf(psf, scalestack):
    """ Calculate scale-dependent moment residuals

    Part of the initialisation for Algorithm 1: lines 12 - 17

    :param psf: psf [nmoments, nx, ny]
    :returns scale-dependent moment psf [nscales, nmoments, nx, ny]
    """
    nmoments2, nx, ny = psf.shape
    nmoments = nmoments2 // 2
    nscales = scalestack.shape[0]
    
    # Lines 12 - 17 from Algorithm 1
    scale_moment_psf = numpy.zeros([nscales, nmoments, nx, ny])
    for t in range(nmoments):
        scale_moment_psf[:, t, ...] = convolve_scalestack(scalestack, psf[t, ...])
    return scale_moment_psf


def calculate_scale_moment_residual(residual, scalestack):
    """ Calculate scale-dependent moment residuals

    Part of the initialisation for Algorithm 1: lines 12 - 17

    :param residual: residual [nmoments, nx, ny]
    :returns scale-dependent moment residual [nscales, nmoments, nx, ny]
    """
    nmoments, nx, ny = residual.shape
    nscales = scalestack.shape[0]
    
    # Lines 12 - 17 from Algorithm 1
    scale_moment_residual = numpy.zeros([nscales, nmoments, nx, ny])
    for t in range(nmoments):
        scale_moment_residual[:, t, ...] = convolve_scalestack(scalestack, residual[t, ...])
    return scale_moment_residual


def calculate_scale_scale_moment_moment_psf(psf, scalestack):
    """ Calculate scale-dependent moment psfs

    Part of the initialisation for Algorithm 1

    :param psf: psf
    :returns scale-dependent moment psf [nscales, nscales, nmoments, nmoments, nx, ny]
    """
    nmoments2, nx, ny = psf.shape
    nmoments = nmoments2 // 2
    nscales = scalestack.shape[0]
    
    # Lines 3 - 5 from Algorithm 1
    scale_scale_moment_moment_psf = numpy.zeros([nscales, nscales, nmoments, nmoments, nx, ny])
    for t in range(nmoments):
        for q in range(nmoments):
            scale_scale_moment_moment_psf[:, :, t, q] = convolve_convolve_scalestack(scalestack, psf[t + q])
    return scale_scale_moment_moment_psf


def calculate_scale_inverse_moment_moment_hessian(scale_scale_moment_moment_psf):
    """Calculate inverse_scale dependent moment moment hessian

    Part of the initialisation for Algorithm 1. Lines 7 - 9

    :param scale_scale_moment_moment_psf: scale_moment_psf [nscales, nscales, nmoments, nmoments]
    :returns: scale-dependent moment-moment inverse hessian
    """
    nscales, _, nmoments, _, nx, ny = scale_scale_moment_moment_psf.shape
    hessian_shape = [nscales, nmoments, nmoments]
    
    scale_moment_moment_hessian = numpy.zeros(hessian_shape)
    scale_inverse_moment_moment_hessian = numpy.zeros(hessian_shape)
    for s in range(nscales):
        scale_moment_moment_hessian[s, ...] = scale_scale_moment_moment_psf[s, s, ..., nx // 2, ny // 2]
        scale_inverse_moment_moment_hessian[s] = numpy.linalg.inv(scale_moment_moment_hessian[s])
    return scale_moment_moment_hessian, scale_inverse_moment_moment_hessian


def calculate_scale_moment_principal_solution(smresidual, ihsmmpsf):
    """ Calculate the principal solution in moment space for each scale

    Lines 20 - 26

    :param smresidual: scale-dependent moment residual [nscales, nmoments, nx, ny]
    :param imhsmmpsf: Inverse of scale dependent moment moment Hessian
    :returns: Decoupled residual images [nscales, nmoments, nx, ny]
    """
    # ihsmmpsf: nscales, nmoments, nmoments
    # smresidual: nscales, nmoments, nx, ny
    smpsol = numpy.einsum("smn,smxy->snxy", ihsmmpsf, smresidual)
    
    return smpsol


def calculate_scale_moment_approximate_principal_solution(smresidual, hsmmpsf):
    """ Calculate approximate principal solution in moment space for each scale

    Equivalent of lines 20 - 26 in ASAPsoft

    :param smresidual: scale-dependent moment residual [nscales, nmoments, nx, ny]
    :param mhsmmpsf: scale dependent moment moment Hessian
    :returns: Decoupled residual images for each scale and moment
    """
    nscales, nmoments, nx, ny = smresidual.shape
    smpsol = numpy.zeros_like(smresidual)
    # We use the inverse of the diagonal elements hessian in moment space to calculate
    # the approximate principal solution in moment space
    for s in range(nscales):
        for moment in range(nmoments):
            smpsol[s, moment] = smresidual[s, moment] / hsmmpsf[s, moment, moment]
    
    return smpsol


def find_optimum_scale_zero_moment(smpsol, windowstack):
    """Find the optimum scale for moment zero
    
    Line 27 of Algorithm 1
    
    :param smpsol: Decoupled residual images for each scale and moment
    :returns: x, y, optimum scale for peak
    """
    nscales, nmoments, nx, ny = smpsol.shape
    sscale = None
    sx = None
    sy = None
    optimum = 0.0
    
    for scale in range(nscales):
        
        if windowstack is not None:
            resid = smpsol[scale, 0, :, :] * windowstack[scale, :, :]
        else:
            resid = smpsol[scale, 0, :, :]
        
        this_max = numpy.max(numpy.abs(resid[scale]))
        if this_max > optimum:
            optimum = this_max
            sscale = scale
            sx, sy = argmax(smpsol[scale, 0, ...])
    
    if sx is None or sy is None or sscale is None:
        raise RuntimeError("find_optimum_scale_zero_moment: Error in finding peak")
    
    return sx, sy, sscale
