# Tim Cornwell <realtimcornwell@gmail.com>
""" Image Deconvolution functions

"""

import warnings

from astropy.convolution import Gaussian2DKernel, convolve
from photutils import fit_2dgaussian

from arl.data.data_models import *
from arl.data.parameters import *
from arl.image.operations import create_image_from_array, copy_image, export_image_to_fits

log = logging.getLogger(__name__)


def deconvolve_cube(dirty: Image, psf: Image, **kwargs):
    """ Clean using a variety of algorithms
    
    Functions that clean a dirty image using a point spread function. The algorithms available are:
    
    - Hogbom CLEAN See: Hogbom CLEAN (1974A&AS...15..417H)
    
    - MultiScale CLEAN See: Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc, 2008 vol. 2 pp. 793-801)
    
    
    :param dirty: Image dirty image
    :param psf: Image Point Spread Function
    :param window: Window image (Bool) - clean where True
    :param algorithm: Cleaning algorithm: 'msclean'|'hogbom'
    :param gain: loop gain (float) 0.7
    :param threshold: Clean threshold (0.0)
    :param fracthres: Fractional threshold (0.01)
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
    :returns: componentimage, residual
    """

    window = get_parameter(kwargs, 'window', None)
    if window == 'quarter':
        qx = dirty.shape[3] // 4
        qy = dirty.shape[2] // 4
        window = numpy.zeros_like(dirty.data)
        window[...,(qy+1):3*qy,(qx+1):3*qx] = 1.0
        log.info('deconvolve_cube: Cleaning inner quarter of each sky plane')
    else:
        window = None
        
    psf_support = get_parameter(kwargs, 'psf_support', dirty.shape[3])
    if isinstance(psf_support, int):
        if (psf_support < psf.shape[2] // 2) and ((psf_support < psf.shape[3] // 2)):
            centre=[psf.shape[2] // 2, psf.shape[3] // 2]
            psf.data = psf.data[...,(centre[0]-psf_support):(centre[0]+psf_support),
                    (centre[1]-psf_support):(centre[1]+psf_support)]
        
    algorithm = get_parameter(kwargs, 'algorithm', 'msclean')
    if algorithm == 'msclean':

        gain = get_parameter(kwargs, 'gain', 0.7)
        assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
        thresh = get_parameter(kwargs, 'threshold', 0.0)
        assert thresh >= 0.0
        niter = get_parameter(kwargs, 'niter', 100)
        assert niter > 0
        scales = get_parameter(kwargs, 'scales', [0, 3, 10, 30])
        fracthresh = get_parameter(kwargs, 'fractional_threshold', 0.01)
        assert 0.0 < fracthresh < 1.0
                    
        comp_array = numpy.zeros(dirty.data.shape)
        residual_array = numpy.zeros(dirty.data.shape)
        for channel in range(dirty.data.shape[0]):
            for pol in range(dirty.data.shape[1]):
                if psf.data[channel, pol, :, :].max():
                    log.info("deconvolve_cube: Processing pol %d, channel %d" % (pol, channel))
                    if window is None:
                        comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                            msclean(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                                    None, gain, thresh, niter, scales, fracthresh)
                    else:
                        comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                            msclean(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                                    window[channel, pol, :, :], gain, thresh, niter, scales, fracthresh)
                else:
                    log.info("deconvolve_cube: Skipping pol %d, channel %d" % (pol, channel))
    elif algorithm == 'hogbom':
        
        gain = get_parameter(kwargs, 'gain', 0.7)
        assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
        thresh = get_parameter(kwargs, 'threshold', 0.0)
        assert thresh >= 0.0
        niter = get_parameter(kwargs, 'niter', 100)
        assert niter > 0
        fracthresh = get_parameter(kwargs, 'fracthresh', 0.01)
        assert 0.0 <= fracthresh < 1.0

        comp_array = numpy.zeros(dirty.data.shape)
        residual_array = numpy.zeros(dirty.data.shape)
        for channel in range(dirty.data.shape[0]):
            for pol in range(dirty.data.shape[1]):
                if psf.data[channel, pol, :, :].max():
                    log.info("deconvolve_cube: Processing pol %d, channel %d" % (pol, channel))
                    if window is None:
                        comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                            hogbom(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                                 None, gain, thresh, niter)
                    else:
                        comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                            hogbom(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                                   window[channel, pol, :, :], gain, thresh, niter)
                else:
                    log.info("deconvolve_cube: Skipping pol %d, channel %d" % (pol, channel))
    else:
        raise ValueError('deconvolve_cube: Unknown algorithm %s' % algorithm)
    
    return create_image_from_array(comp_array, dirty.wcs), create_image_from_array(residual_array, dirty.wcs)


def deconvolve_mfs(dirty: Image, psf: Image, **kwargs):
    """ MFS Clean using a variety of algorithms

    Functions that clean a dirty image using a point spread function. The algorithms available are:

    - Hogbom CLEAN See: Hogbom CLEAN (1974A&AS...15..417H)

    - MultiScale CLEAN See: Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc, 2008 vol. 2 pp. 793-801)


    :param dirty: Image dirty image
    :param psf: Image Point Spread Function
    :param params: 'algorithm': 'msclean'|'hogbom', 'gain': loop gain (float)
    :returns: componentimage, residual
    """
    
    raise ValueError("deconvolve_mfs: not yet implemented")


def restore_mfs(dirty: Image, clean: Image, psf: Image, **kwargs):
    """ Restore an MFS clean image

    :param dirty:
    :param clean: Image clean model (i.e. no smoothing)
    :param psf: Image Point Spread Function
    :param params: 'algorithm': 'msclean'|'hogbom', 'gain': loop gain (float)
    :returns: restored image
    """
    
    raise ValueError("restore_mfs: not yet implemented")



def overlapIndices(res, psf, peakx, peaky):
    """ Find the indices where two arrays overlap

    :param a1: First array
    :param a2: Second array
    :param shiftx: Shift in x applied to a1
    :param shifty: Shift in y applied to a2
    :returns (limits in a1, limits in a2)
    """
    nx, ny = res.shape[0], res.shape[1]
    psfwidthx, psfwidthy = psf.shape[0] // 2, psf.shape[1] // 2
    psfpeakx, psfpeaky = psf.shape[0] // 2, psf.shape[1] // 2
    # Step 1 line up the coordinate ignoring limits
    res_lower = (max(0, peakx - psfwidthx),  max(0, peaky - psfwidthy))
    res_upper = (min(nx, peakx + psfwidthx), min(peaky + psfwidthy, ny))
    psf_lower = (max(0, psfpeakx + (res_lower[0]-peakx)), max(0, psfpeaky + (res_lower[1]-peaky)))
    psf_upper = (min(psf.shape[0], psfpeakx + (res_upper[0]-peakx)), min(psfpeaky + (res_upper[1]-peaky), psf.shape[1]))

    return (res_lower[0], res_upper[0], res_lower[1], res_upper[1]), \
           (psf_lower[0], psf_upper[0], psf_lower[1], psf_upper[1])


def argmax(a):
    """ Return unravelled index of the maximum

    param: a: array to be searched
    """
    return numpy.unravel_index(a.argmax(), a.shape)


def hogbom(dirty,
           psf,
           window,
           gain,
           thresh,
           niter):
    """
    Hogbom CLEAN (1974A&AS...15..417H)

    :param dirty: The dirty Image, i.e., the Image to be deconvolved
    :param psf: The point spread-function
    :param window: Regions where clean components are allowed. If True, all of the dirty Image is assumed to be
    allowed for clean components
    :param gain: The "loop gain", i.e., the fraction of the brightest pixel that is removed in each iteration
    :param thresh: Cleaning stops when the maximum of the absolute deviation of the residual is less than this value
    :param niter: Maximum number of components to make if the threshold `thresh` is not hit
    :returns: clean Skycomponent Image, residual Image
    """
    
    assert 0.0 < gain < 2.0
    assert niter > 0
    
    comps = numpy.zeros(dirty.shape)
    res = numpy.array(dirty)
    pmax = psf.max()
    assert pmax > 0.0
    psfpeak = argmax(numpy.fabs(psf))
    if window is True:
        window = 1.0
    for i in range(niter):
        if window is not None:
            mx, my = numpy.unravel_index((numpy.fabs(res*window)).argmax(), dirty.shape)
        else:
            mx, my = numpy.unravel_index((numpy.fabs(res)).argmax(), dirty.shape)
        mval = res[mx, my] * gain / pmax
        comps[mx, my] += mval
        a1o, a2o = overlapIndices(dirty, psf, mx, my)
        res[a1o[0]:a1o[1], a1o[2]:a1o[3]] -= psf[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval
        if i % (niter // 10) == 0:
            log.info("hogbom: Minor cycle %d, peak %s at [%d, %d]" % (i, res[mx, my], mx, my))
        if numpy.fabs(res).max() < thresh:
            break
    return comps, res


def msclean(dirty,
            psf,
            window,
            gain,
            thresh,
            niter,
            scales,
            fracthresh,
            **kwargs):
    """ Perform multiscale clean

    Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc, 2008 vol. 2 pp. 793-801)

    :param fracthresh:
    :param dirty: The dirty Image, i.e., the Image to be deconvolved
    :param psf: The point spread-function
    :param window: Regions where clean components are allowed. If
    True, then all of the dirty Image is assumed to be allowed for
    clean components
    :param gain: The "loop gain", i.e., the fraction of the brightest
    pixel that is removed in each iteration
    :param thresh: Cleaning stops when the maximum of the absolute
    deviation of the residual is less than this value
    :param niter: Maximum number of components to make if the
    threshold "thresh" is not hit
    :param scales: Scales (in pixels width) to be used
    :returns: clean component Image, residual Image
    """
    assert 0.0 < gain < 2.0
    assert niter > 0
    assert len(scales) > 0
    
    comps = numpy.zeros(dirty.shape)
    
    pmax = psf.max()
    assert pmax > 0.0
    
    psfpeak = argmax(numpy.fabs(psf))
    log.info("msclean: Peak of PSF = %s at %s" % (pmax, psfpeak))
    dmax = dirty.max()
    dpeak = argmax(dirty)
    log.info("msclean: Peak of Dirty = %s at %s" % (dmax, dpeak))
    lpsf = psf / pmax
    ldirty = dirty / pmax
    
    # Create the scale images and form all the various products we need. We
    # use a third dimension to hold the scale-related images. scalestack is a 3D
    # cube holding the different scale images. convolvestack will take a 2D Image
    # and add a third dimension holding the scale-convolved versions.

    scaleshape = [len(scales), ldirty.shape[0], ldirty.shape[1]]
    scalestack = createscalestack(scaleshape, scales, norm=True)

    pscaleshape = [len(scales), lpsf.shape[0], lpsf.shape[1]]
    pscalescaleshape = [len(scales), len(scales), lpsf.shape[0], lpsf.shape[1]]
    pscalestack = createscalestack(pscaleshape, scales, norm=True)

    psf_scalestack = convolvescalestack(pscalestack, numpy.array(lpsf))
    res_scalestack = convolvescalestack(scalestack, numpy.array(ldirty))

    # Evaluate the coupling matrix between the various scale sizes.
    psf_scalescalestack = numpy.zeros(pscalescaleshape)
    for iscale in numpy.arange(len(scales)):
        psf_scalescalestack[ :, iscale, :, :] = convolvescalestack(pscalestack, psf_scalestack[iscale, :, :])
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
        windowstack=numpy.zeros_like(scalestack)
        windowstack[convolvescalestack(scalestack, window)>0.9]=1.0
    
    log.info("msclean: Max abs in dirty Image = %.6f" % numpy.fabs(res_scalestack[0, :, :]).max())
    absolutethresh = max(thresh, fracthresh * numpy.fabs(res_scalestack[0, :, :]).max())
    log.info("msclean: Start of minor cycle")
    log.info("msclean: This minor cycle will stop at %d iterations or peak < %s" % (niter, absolutethresh))
    
    for i in range(niter):
        # Find peak over all smoothed images
        mx, my, mscale = findabsmaxstack(res_scalestack, windowstack, coupling_matrix)
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


def createscalestack(scaleshape, scales, norm=True):
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
            # Unroll this since sphfn needs a scalar
            for y in range(ycen - halfscale - 1, ycen + halfscale + 1):
                for x in range(xcen - halfscale - 1, xcen + halfscale + 1):
                    fx = float(x - xcen)
                    fy = float(y - ycen)
                    r2 = rscale2 * (fx * fx + fy * fy)
                    r = numpy.sqrt(r2)
                    basis[iscale, x, y] = sphfn(r) * (1.0 - r ** 2)
            basis[basis < 0.0] = 0.0
            if norm:
                basis[iscale, :, :] /= numpy.sum(basis[iscale, :, :])
        else:
            basis[iscale, xcen, ycen] = 1.0
    return basis


def convolvescalestack(scalestack, img):
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


def findabsmaxstack(stack, windowstack, couplingmatrix):
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
            resid = stack[iscale, :, :] * windowstack[iscale, : , :]
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


def sphfn(vnu):
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


def restore_cube(model, psf, residual=None):
    """ Restore the model image to the residuals

    :params psf: Input PSF
    :returns: major axis, minor axis, position angle (in radians)

    """
    restored = copy_image(model)
    
    npixel = psf.data.shape[3]
    sl = slice(npixel // 2 - 7, npixel // 2 + 8)
    
    # isotropic at the moment!
    try:
        fit = fit_2dgaussian(psf.data[0, 0, sl, sl])
        if fit.x_stddev <= 0.0 or fit.y_stddev <= 0.0:
            log.debug('restore_cube: error in fitting to psf, using 1 pixel stddev')
            size = 1.0
        else:
            size = max(fit.x_stddev, fit.y_stddev)
            log.debug('restore_cube: psfwidth = %s' %(size))
    except:
        log.debug('restore_cube: warning in fit to psf, using 1 pixel stddev')
        size = 1.0
    
    # By convention, we normalise the peak not the integral so this is the volume of the Gaussian
    norm = 2.0 * numpy.pi * size ** 2
    gk = Gaussian2DKernel(size)
    for chan in range(model.shape[0]):
        for pol in range(model.shape[1]):
            restored.data[chan, pol, :, :] = norm * convolve(model.data[chan, pol, :, :], gk, normalize_kernel=False)
    if residual is not None:
        restored.data += residual.data
    return restored
