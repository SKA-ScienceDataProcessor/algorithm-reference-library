# Tim Cornwell <realtimcornwell@gmail.com>
""" Image Deconvolution functions

"""
import logging

import numpy

from arl.image.hogbom import argmax, overlapIndices
from arl.image.msclean import convolve_scalestack, convolve_convolve_scalestack, create_scalestack

log = logging.getLogger(__name__)


def msmfsclean(dirty, psf, window, gain, thresh, niter, scales, fracthresh):
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
    log.info("msclean: Peak of PSF = %s at %s" % (pmax, psfpeak))
    dmax = dirty.max()
    dpeak = argmax(dirty)
    log.info("msclean: Peak of Dirty = %s at %s" % (dmax, dpeak))
    lpsf = psf / pmax
    ldirty = dirty / pmax
    
    nmoments, ny, nx = dirty.shape
    assert psf.shape[0] == 2 * nmoments
    
    # Create the "scale basis functions" in Algorithm 1
    scaleshape = [nscales, ldirty.shape[1], ldirty.shape[2]]
    scalestack = create_scalestack(scaleshape, scales, norm=True)
    
    # Calculate scaled convolved psfs
    smpsf = calculate_scale_moment_psf(lpsf, scalestack)
    
    # Calculate scale moment psf
    ssmmpsf = calculate_scale_scale_moment_moment_psf(lpsf, scalestack)
    ihsmmpsf = calculate_scale_inverse_moment_moment_hessian(ssmmpsf)
    
    # Calculate scale convolutions of moment residuals
    smresidual = calculate_scale_moment_residual(ldirty, scalestack)
    
    log.info("msclean: Coupling matrix =\n %s" % ihsmmpsf)
    
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
    for i in range(niter):
        
        # Calculate the principal solution in moment-moment axes. This decouples the moments
        smpsol = calculate_scale_moment_principal_solution(smresidual, ihsmmpsf)
        
        # Now find the location and scale for which the zero'th moment is maximised
        mx, my, mscale = find_optimum_scale_zero_moment(smpsol, windowstack)
        
        if mx is None or my is None or mscale is None:
            raise RuntimeError("msmfsclean: Error in finding peak")
        
        # Find the values to subtract in moment spaces
        mval = smpsol[mscale, :, mx, my]
        raw_mval = smresidual[mscale, :, mx, my]
        
        # Report on progress
        if i % (niter // 10) == 0:
            log.info("msclean: Minor cycle %d, peak %s at [%d, %d, %d]" % (i, smresidual[0, :, mx, my], mx, my, mscale))
        
        # Are we ready to stop yet?
        if numpy.fabs(mval[0]) < absolutethresh:
            log.info("msclean: Absolute value of peak %.6f is below stopping threshold %.6f" \
                     % (numpy.fabs(smresidual[mscale, 0, mx, my]), absolutethresh))
            break
        
        # Calculate indices needed for no shift and shifted
        no_shift, shifted = overlapIndices(ldirty[0, ...], psf[0, ...], mx, my)
        
        # Update model and residual image
        m_model = update_moment_model(m_model, scalestack, no_shift, shifted, gain, mscale, mval)
        smresidual = update_scale_moment_residual(smresidual, ssmmpsf, no_shift, shifted, gain, mscale, mval)
    
    log.info("msclean: End of minor cycles")
    
    return m_model, pmax * smresidual[0, :, :, :]


def update_scale_moment_residual(smresidual, ssmmpsf, a1o, a2o, gain, mscale, mval):
    """ Update residual by subtracting the effect of model update for each moment
    
    """
    # Lines 30 - 32 of Algorithm 1.
    nscales, nmoments, _, _ = smresidual.shape
    for t in range(nmoments):
        for q in range(nmoments):
            for s in range(nscales):
                smresidual[s, t, a1o[0]:a1o[1], a1o[2]:a1o[3]] -= \
                    ssmmpsf[mscale, s, t, q, a2o[0]:a2o[1], a2o[2]:a2o[3]] * gain * mval[t]
    
    return smresidual


def update_moment_model(m_model, scalestack, a1o, a2o, gain, mscale, mval):
    """Update model with an appropriately scaled and centered blob for each moment
    
    """
    # Lines 28 - 33 of Algorithm 1
    nmoments, _, _ = m_model.shape
    for t in range(nmoments):
        # Line 29 of Algorithm 1. Note that the convolution is implemented here as an
        # appropriate shift.
        m_model[t, a1o[0]:a1o[1], a1o[2]:a1o[3]] += \
            scalestack[mscale, a2o[0]:a2o[1], a2o[2]:a2o[3]] * gain * mval[t]
    
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
    :returns: scale-dependent momemt-moment inverse hessian
    """
    nmoments, _, nscales, _, nx, ny = scale_scale_moment_moment_psf.shape
    scale_momemt_moment_hessian = scale_scale_moment_moment_psf[..., nx // 2, ny // 2]
    scale_inverse_momemt_moment_hessian = numpy.zeros_like(scale_momemt_moment_hessian)
    for s in range(nscales):
        scale_inverse_momemt_moment_hessian[s, ...] = \
            numpy.linalg.inv(scale_momemt_moment_hessian[s, ...])
    return scale_inverse_momemt_moment_hessian


def calculate_scale_moment_principal_solution(smresidual, ihsmmpsf):
    """ Calculate the principal solution in moment space for each scale
    
    Lines 20 - 26
    
    :param smresidual: scale-dependent moment residual [nscales, nmoments, nx, ny]
    :param imhsmmpsf: Inverse of scale dependent moment moment Hessian
    :returns: Decoupled residual images for each scale and moment
    """
    nscales, nmoments, nx, ny = smresidual.shape
    smpsol = numpy.zeros_like(smresidual)
    # We use the inverse hessian in moment space to calculate the principal solution in moment space
    for s in range(nscales):
        for ix in range(nx):
            for iy in range(ny):
                smpsol[s, :, ix, iy] = numpy.dot(ihsmmpsf[s, s, ...], smresidual[s, :, ix, iy])
    
    return smpsol


def find_optimum_scale_zero_moment(smpsol, windowstack):
    """Find the optimum scale for moment zero
    
    Line 27 of Algorithm 1
    
    :param smpsol: Decoupled residual images for each scale and moment
    :returns: x, y, optimum scale for peak
    """
    nscales, nmoments, nx, ny = smpsol.shape
    sscale = 0
    sx = 0
    sy = 0
    optimum = 0.0
    
    for scale in range(nscales):
        
        if windowstack is not None:
            resid = smpsol[scale, 0, :, :] * windowstack[scale, :, :]
        else:
            resid = smpsol[scale, :, :]
        
        this_max = numpy.max(numpy.abs(resid))
        if this_max > optimum:
            optimum = this_max
            sscale = scale
            sx, sy = argmax(smpsol[scale, 0, ...])
    
    return sx, sy, sscale
