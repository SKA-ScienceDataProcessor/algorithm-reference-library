# Tim Cornwell <realtimcornwell@gmail.com>)

import matplotlib.pyplot as plt
import numpy as np

from crocodile.clean import overlapIndices, argmax
from crocodile.synthesis import sortw, doimg, dopredict, simplepredict, simpleimg

def msclean(dirty,
            psf,
            window,
            gain,
            thresh,
            niter,
            scales,
            fracthresh,
            params={}):
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

    comps = np.zeros(dirty.shape)

    pmax = psf.max()
    assert pmax > 0.0

    psfpeak = argmax(np.fabs(psf))
    print("Peak of PSF = %s at %s" % (pmax, psfpeak))
    dmax = dirty.max()
    dpeak = argmax(dirty)
    print("Peak of Dirty = %s at %s" % (dmax, dpeak))
    lpsf = psf / pmax
    ldirty = dirty / pmax

    # Create the scale images and form all the various products we need. We
    # use a third dimension to hold the scale-related images. scalestack is a 3D
    # cube holding the different scale images. convolvestack will take a 2D Image
    # and add a third dimension holding the scale-convolved versions.

    scaleshape = [ldirty.shape[0], ldirty.shape[1], len(scales)]
    scalescaleshape = [ldirty.shape[0], ldirty.shape[1], len(scales), len(scales)]
    scalestack = createscalestack(scaleshape, scales, norm=True)

    couplingMatrix = np.zeros([len(scales), len(scales)])
    psfscalestack = convolvescalestack(scalestack, np.array(lpsf))
    resscalestack = convolvescalestack(scalestack, np.array(ldirty))
    # Evaluate the coupling matrix between the various scale sizes.
    psfscalescalestack = np.zeros(scalescaleshape)
    for iscale in np.arange(len(scales)):
        psfscalescalestack[:, :, :, iscale] = convolvescalestack(scalestack, psfscalestack[:, :, iscale])
        psfscalescalestack[:, :, iscale, :] = psfscalescalestack[:, :, :, iscale]
    for iscale in np.arange(len(scales)):
        for iscale1 in np.arange(len(scales)):
            couplingMatrix[iscale, iscale1] = np.max(psfscalescalestack[:, :, iscale, iscale1])
    print("Coupling matrix =\n %s" % couplingMatrix)

    # The window is scale dependent - we form it by smoothing and thresholding
    # the input window. This prevents components being placed too close to the
    # edge of the Image.

    if window is True:
        windowstack = np.ones(scalestack.shape, np.bool)
    #    windowstack=convolvescalestack(scalestack, window)>0.9
    window = np.ones(scalestack.shape, np.bool)

    """ The minor cycle
    """
    print("Max abs in dirty Image = %.6f" % np.fabs(resscalestack[:, :, 0]).max())
    absolutethresh = max(thresh, fracthresh * np.fabs(resscalestack[:, :, 0]).max())
    print("Start of minor cycle")
    print("This minor cycle will stop at %d iterations or peak < %s" % (niter, absolutethresh))

    for i in range(niter):
        # Find peak over all smoothed images
        mx, my, mscale = findabsmaxstack(resscalestack, window, couplingMatrix)
        if mx is None or my is None or mscale is None:
            print("Error in finding peak")
            break

        # Find the values to subtract, accounting for the coupling matrix
        mval = np.zeros(len(scales))
        mval[mscale] = resscalestack[mx, my, mscale] / couplingMatrix[mscale, mscale]
        if i % 10 == 0:
            print("Minor cycle %d, peak %s at [%d, %d, %d]" % \
                  (i, resscalestack[mx, my, :], mx, my, mscale))
        if np.fabs(mval[mscale]) < absolutethresh:
            print("Absolute value of peak %.6f is below stopping threshold %.6f" \
                  % (np.fabs(resscalestack[mx, my, mscale]), absolutethresh))
            break

        # Update the cached residuals and add to the cached model.
        a1o, a2o = overlapIndices(dirty, psf, mx - psfpeak[0], my - psfpeak[1])
        if np.abs(mval[mscale]) > 0:
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
    print("End of minor cycles")
    return comps, pmax * resscalestack[:, :, 0]


def createscalestack(scaleshape, scales, norm=True):
    """ Create a cube consisting of the scales

    :param scaleshape: desired shape of stack
    :param scales: scales (in pixels)
    :param norm: Normalise each plane to unity?
    :returns: stack
    """
    assert scaleshape[2] == len(scales)

    basis = np.zeros(scaleshape)
    nx = scaleshape[0]
    ny = scaleshape[1]
    xcen = int(np.ceil(float(nx) / 2.0))
    ycen = int(np.ceil(float(ny) / 2.0))
    for iscale in np.arange(0, len(scales)):
        halfscale = int(np.ceil(scales[iscale] / 2.0))
        if scales[iscale] > 0.0:
            rscale2 = 1.0 / (float(scales[iscale]) / 2.0) ** 2
            x = range(xcen - halfscale - 1, xcen + halfscale + 1)
            fx = np.array(x, 'float') - float(xcen)
            # Unroll this since sphfn needs a scalar
            for y in range(ycen - halfscale - 1, ycen + halfscale + 1):
                for x in range(xcen - halfscale - 1, xcen + halfscale + 1):
                    fx = float(x - xcen)
                    fy = float(y - ycen)
                    r2 = rscale2 * (fx * fx + fy * fy)
                    r = np.sqrt(r2)
                    basis[x, y, iscale] = sphfn(r) * (1.0 - r ** 2)
            basis[basis < 0.0] = 0.0
            if norm:
                basis[:, :, iscale] /= np.sum(basis[:, :, iscale])
        else:
            basis[xcen, ycen, iscale] = 1.0
    return basis


def convolvescalestack(scalestack, img):
    """Convolve img by the specified scalestack, returning the resulting stack

    :param scalestack: stack containing the scales
    :param img: Image to be convolved
    :returns: stack
    """

    convolved = np.zeros(scalestack.shape)
    ximg = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img)))

    nscales = scalestack.shape[2]
    for iscale in range(nscales):
        xscale = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(scalestack[:, :, iscale])))
        xmult = ximg * xscale
        convolved[:, :, iscale] = np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(xmult))))
    return convolved


def findabsmaxstack(stack, window, couplingmatrix):
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
        mx, my = np.unravel_index(np.fabs(stack[:, :, iscale]).argmax(), pshape)
        thisabsmax = stack[mx, my, iscale] / couplingmatrix[iscale, iscale]
        if abs(thisabsmax) > abs(pabsmax):
            px = mx
            py = my
            pscale = iscale
            pabsmax = stack[px, py, pscale] / couplingmatrix[iscale, iscale]
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

    p = np.zeros((2, 5))
    q = np.zeros((2, 3))

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

    if value < 0.: value = 0.

    return value
