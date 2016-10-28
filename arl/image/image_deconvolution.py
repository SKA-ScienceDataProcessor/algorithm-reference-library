# Tim Cornwell <realtimcornwell@gmail.com>
""" Image Deconvolution functions

"""

from data.data_models import *
from data.parameters import *
from image.image_operations import create_image_from_array

log = logging.getLogger("arl.image_deconvolution")


def deconvolve_cube(dirty: Image, psf: Image, params=None):
    """ Clean using a variety of algorithms
    
    Functions that clean a dirty image using a point spread function. The algorithms available are:
    
    - Hogbom CLEAN See: Hogbom CLEAN (1974A&AS...15..417H)
    
    - MultiScale CLEAN See: Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc, 2008 vol. 2 pp. 793-801)
    
    
    :param dirty: Image dirty image
    :param psf: Image Point Spread Function
    :param params: 'algorithm': 'msclean'|'hogbom', 'gain': loop gain (float)
    :returns: componentimage, residual
    """
    if params is None:
        params = {}
    log_parameters(params)
    algorithm = get_parameter(params, 'algorithm', 'msclean')
    if algorithm == 'msclean':
        
        window = get_parameter(params, 'window', None)
        gain = get_parameter(params, 'gain', 0.7)
        assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
        thresh = get_parameter(params, 'threshold', 0.0)
        assert thresh >= 0.0
        niter = get_parameter(params, 'niter', 100)
        assert niter > 0
        scales = get_parameter(params, 'scales', [0, 3, 10, 30])
        fracthresh = get_parameter(params, 'fracthresh', 0.01)
        assert 0.0 < fracthresh < 1.0
        
        comp_array = numpy.zeros(dirty.data.shape)
        residual_array = numpy.zeros(dirty.data.shape)
        for channel in range(dirty.data.shape[0]):
            for pol in range(dirty.data.shape[1]):
                if psf.data[channel, pol, :, :].max():
                    log.debug("deconvolve_cube: Processing pol %d, channel %d" % (pol, channel))
                    comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                        _msclean(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                                 window, gain, thresh, niter, scales, fracthresh)
                else:
                    log.debug("deconvolve_cube: Skipping pol %d, channel %d" % (pol, channel))
    elif algorithm == 'hogbom':
        
        window = get_parameter(params, 'window', None)
        gain = get_parameter(params, 'gain', 0.7)
        assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
        thresh = get_parameter(params, 'threshold', 0.0)
        assert thresh > 0.0
        niter = get_parameter(params, 'niter', 100)
        assert niter > 0
        fracthresh = get_parameter(params, 'fracthresh', 0.01)
        assert 0.0 < fracthresh < 1.0
        
        comp_array = numpy.zeros(dirty.data.shape)
        residual_array = numpy.zeros(dirty.data.shape)
        for channel in range(dirty.data.shape[0]):
            for pol in range(dirty.data.shape[1]):
                if psf.data[channel, pol, :, :].max():
                    log.debug("deconvolve_cube: Processing pol %d, channel %d" % (pol, channel))
                    comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                        _hogbom(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                                window, gain, thresh, niter)
                else:
                    log.debug("deconvolve_cube: Skipping pol %d, channel %d" % (pol, channel))
    else:
        raise ValueError('deconvolve_cube: Unknown algorithm %s' % algorithm)
    
    return create_image_from_array(comp_array, dirty.wcs), create_image_from_array(residual_array, dirty.wcs)


def restore_cube(dirty: Image, clean: Image, psf: Image, params=None):
    """ Restore a clean image

    :param dirty:
    :param clean: Image clean model (i.e. no smoothing)
    :param psf: Image Point Spread Function
    :param params: 'algorithm': 'msclean'|'hogbom', 'gain': loop gain (float)
    :returns: restored image
    """
    if params is None:
        params = {}
    log_parameters(params)
    log.error("restore_image: not yet implemented")
    return Image()


def deconvolve_mfs(dirty: Image, psf: Image, params=None):
    """ MFS Clean using a variety of algorithms

    Functions that clean a dirty image using a point spread function. The algorithms available are:

    - Hogbom CLEAN See: Hogbom CLEAN (1974A&AS...15..417H)

    - MultiScale CLEAN See: Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc, 2008 vol. 2 pp. 793-801)


    :param dirty: Image dirty image
    :param psf: Image Point Spread Function
    :param params: 'algorithm': 'msclean'|'hogbom', 'gain': loop gain (float)
    :returns: componentimage, residual
    """
    if params is None:
        params = {}
    log_parameters(params)
    log.error("deconvolve_mfs: not yet implemented")
    return Image()


def restore_mfs(dirty: Image, clean: Image, psf: Image, params=None):
    """ Restore an MFS clean image

    :param dirty:
    :param clean: Image clean model (i.e. no smoothing)
    :param psf: Image Point Spread Function
    :param params: 'algorithm': 'msclean'|'hogbom', 'gain': loop gain (float)
    :returns: restored image
    """
    if params is None:
        params = {}
    log_parameters(params)
    log.error("restore_mfs: not yet implemented")
    return Image()


def _overlapIndices(a1, a2,
                    shiftx, shifty):
    """ Find the indices where two arrays overlapIndices

    :param a1: First array
    :param a2: Second array
    :param shiftx: Shift in x applied to a1
    :param shifty: Shift in y applied to a2
    """
    if shiftx >= 0:
        a1xbeg = shiftx
        a2xbeg = 0
        a1xend = a1.shape[0]
        a2xend = a1.shape[0] - shiftx
    else:
        a1xbeg = 0
        a2xbeg = -shiftx
        a1xend = a1.shape[0] + shiftx
        a2xend = a1.shape[0]
    
    if shifty >= 0:
        a1ybeg = shifty
        a2ybeg = 0
        a1yend = a1.shape[1]
        a2yend = a1.shape[1] - shifty
    else:
        a1ybeg = 0
        a2ybeg = -shifty
        a1yend = a1.shape[1] + shifty
        a2yend = a1.shape[1]
    
    return (a1xbeg, a1xend, a1ybeg, a1yend), (a2xbeg, a2xend, a2ybeg, a2yend)


def _argmax(a):
    """ Return unravelled index of the maximum

    param: a: array to be searched
    """
    return numpy.unravel_index(a.argmax(), a.shape)


def _hogbom(dirty,
            psf,
            window,
            gain,
            thresh,
            niter,
            params=None):
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
    if params is None:
        params = {}
    log_parameters(params)
    
    assert 0.0 < gain < 2.0
    assert niter > 0
    
    comps = numpy.zeros(dirty.shape)
    res = numpy.array(dirty)
    pmax = psf.max()
    assert pmax > 0.0
    psfpeak = _argmax(numpy.fabs(psf))
    if window is True:
        window = numpy.ones(dirty.shape, numpy.bool)
    for i in range(niter):
        mx, my = numpy.unravel_index((numpy.fabs(res[window])).argmax(), dirty.shape)
        mval = res[mx, my] * gain / pmax
        comps[mx, my] += mval
        a1o, a2o = _overlapIndices(dirty, psf,
                                   mx - psfpeak[0],
                                   my - psfpeak[1])
        res[a1o[0]:a1o[1], a1o[2]:a1o[3]] -= psf[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval
        if numpy.fabs(res).max() < thresh:
            break
    return comps, res


def _msclean(dirty,
             psf,
             window,
             gain,
             thresh,
             niter,
             scales,
             fracthresh,
             params=None):
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
    if params is None:
        params = {}
    log_parameters(params)
    assert 0.0 < gain < 2.0
    assert niter > 0
    assert len(scales) > 0
    
    comps = numpy.zeros(dirty.shape)
    
    pmax = psf.max()
    assert pmax > 0.0
    
    psfpeak = _argmax(numpy.fabs(psf))
    log.info("msclean: Peak of PSF = %s at %s" % (pmax, psfpeak))
    dmax = dirty.max()
    dpeak = _argmax(dirty)
    log.info("msclean: Peak of Dirty = %s at %s" % (dmax, dpeak))
    lpsf = psf / pmax
    ldirty = dirty / pmax
    
    # Create the scale images and form all the various products we need. We
    # use a third dimension to hold the scale-related images. scalestack is a 3D
    # cube holding the different scale images. convolvestack will take a 2D Image
    # and add a third dimension holding the scale-convolved versions.
    
    scaleshape = [ldirty.shape[0], ldirty.shape[1], len(scales)]
    scalescaleshape = [ldirty.shape[0], ldirty.shape[1], len(scales), len(scales)]
    scalestack = _createscalestack(scaleshape, scales, norm=True)
    
    coupling_matrix = numpy.zeros([len(scales), len(scales)])
    psfscalestack = _convolvescalestack(scalestack, numpy.array(lpsf))
    resscalestack = _convolvescalestack(scalestack, numpy.array(ldirty))
    # Evaluate the coupling matrix between the various scale sizes.
    psfscalescalestack = numpy.zeros(scalescaleshape)
    for iscale in numpy.arange(len(scales)):
        psfscalescalestack[:, :, :, iscale] = _convolvescalestack(scalestack, psfscalestack[:, :, iscale])
        psfscalescalestack[:, :, iscale, :] = psfscalescalestack[:, :, :, iscale]
    for iscale in numpy.arange(len(scales)):
        for iscale1 in numpy.arange(len(scales)):
            coupling_matrix[iscale, iscale1] = numpy.max(psfscalescalestack[:, :, iscale, iscale1])
    log.info("msclean: Coupling matrix =\n %s" % coupling_matrix)
    
    # The window is scale dependent - we form it by smoothing and thresholding
    # the input window. This prevents components being placed too close to the
    # edge of the Image.
    
    if window is True:
        windowstack = numpy.ones(scalestack.shape, numpy.bool)
    # windowstack=convolvescalestack(scalestack, window)>0.9
    window = numpy.ones(scalestack.shape, numpy.bool)
    
    log.info("msclean: Max abs in dirty Image = %.6f" % numpy.fabs(resscalestack[:, :, 0]).max())
    absolutethresh = max(thresh, fracthresh * numpy.fabs(resscalestack[:, :, 0]).max())
    log.info("msclean: Start of minor cycle")
    log.info("msclean: This minor cycle will stop at %d iterations or peak < %s" % (niter, absolutethresh))
    
    for i in range(niter):
        # Find peak over all smoothed images
        mx, my, mscale = _findabsmaxstack(resscalestack, window, coupling_matrix)
        if mx is None or my is None or mscale is None:
            log.warning("msclean: Error in finding peak")
            break
        
        # Find the values to subtract, accounting for the coupling matrix
        mval = numpy.zeros(len(scales))
        mval[mscale] = resscalestack[mx, my, mscale] / coupling_matrix[mscale, mscale]
        if i % 10 == 0:
            log.info("msclean: Minor cycle %d, peak %s at [%d, %d, %d]" % \
                     (i, resscalestack[mx, my, :], mx, my, mscale))
        if numpy.fabs(mval[mscale]) < absolutethresh:
            log.info("msclean: Absolute value of peak %.6f is below stopping threshold %.6f" \
                     % (numpy.fabs(resscalestack[mx, my, mscale]), absolutethresh))
            break
        
        # Update the cached residuals and add to the cached model.
        a1o, a2o = _overlapIndices(dirty, psf, mx - psfpeak[0], my - psfpeak[1])
        if numpy.abs(mval[mscale]) > 0:
            # Cross subtract from other scales
            for iscale in range(len(scales)):
                resscalestack[a1o[0]:a1o[1], a1o[2]:a1o[3], iscale] -= \
                    psfscalescalestack[a2o[0]:a2o[1], a2o[2]:a2o[3], iscale, mscale] * \
                    gain * mval[mscale]
            comps[a1o[0]:a1o[1], a1o[2]:a1o[3]] += \
                scalestack[a2o[0]:a2o[1], a2o[2]:a2o[3], mscale] * \
                gain * mval[mscale]
        else:
            break
    log.info("msclean: End of minor cycles")
    return comps, pmax * resscalestack[:, :, 0]


def _createscalestack(scaleshape, scales, norm=True):
    """ Create a cube consisting of the scales

    :param scaleshape: desired shape of stack
    :param scales: scales (in pixels)
    :param norm: Normalise each plane to unity?
    :returns: stack
    """
    assert scaleshape[2] == len(scales)
    
    basis = numpy.zeros(scaleshape)
    nx = scaleshape[0]
    ny = scaleshape[1]
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
                    basis[x, y, iscale] = _sphfn(r) * (1.0 - r ** 2)
            basis[basis < 0.0] = 0.0
            if norm:
                basis[:, :, iscale] /= numpy.sum(basis[:, :, iscale])
        else:
            basis[xcen, ycen, iscale] = 1.0
    return basis


def _convolvescalestack(scalestack, img):
    """Convolve img by the specified scalestack, returning the resulting stack

    :param scalestack: stack containing the scales
    :param img: Image to be convolved
    :returns: stack
    """
    
    convolved = numpy.zeros(scalestack.shape)
    ximg = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(img)))
    
    nscales = scalestack.shape[2]
    for iscale in range(nscales):
        xscale = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(scalestack[:, :, iscale])))
        xmult = ximg * xscale
        convolved[:, :, iscale] = numpy.real(numpy.fft.ifftshift(numpy.fft.ifft2(numpy.fft.ifftshift(xmult))))
    return convolved


def _findabsmaxstack(stack, window, couplingmatrix):
    """Find the location and value of the absolute maximum in this stack
    :param stack: stack to be searched
    :param window: Window for the searched
    :param couplingmatrix: Coupling matrix between difference scales
    :returns: x, y, scale

    """
    pabsmax = 0.0
    pscale = None
    px = None
    py = None
    pshape = [stack.shape[0], stack.shape[1]]
    for iscale in range(stack.shape[2]):
        mx, my = numpy.unravel_index(numpy.fabs(stack[:, :, iscale]).argmax(), pshape)
        thisabsmax = stack[mx, my, iscale] / couplingmatrix[iscale, iscale]
        if abs(thisabsmax) > abs(pabsmax):
            px = mx
            py = my
            pscale = iscale
            pabsmax = stack[px, py, pscale] / couplingmatrix[iscale, iscale]
    return px, py, pscale


def _sphfn(vnu):
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
