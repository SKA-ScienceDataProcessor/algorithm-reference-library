
""" Image Deconvolution functions

"""

import numpy
import logging
log = logging.getLogger(__name__)

def hogbom(dirty, psf, window, gain, thresh, niter, fracthresh):
    """ Clean the point spread function from a dirty image
    
    See Hogbom CLEAN (1974A&AS...15..417H)

    This version operates on numpy arrays.

    :param dirty: The dirty Image, i.e., the Image to be deconvolved
    :param psf: The point spread-function
    :param window: Regions where clean components are allowed. If True, all of the dirty Image is assumed to be
    allowed for clean components
    :param gain: The "loop gain", i.e., the fraction of the brightest pixel that is removed in each iteration
    :param thresh: Cleaning stops when the maximum of the absolute deviation of the residual is less than this value
    :param niter: Maximum number of components to make if the threshold `thresh` is not hit
    :returns: clean component Image, residual Image
    """
    
    assert 0.0 < gain < 2.0
    assert niter > 0
    
    log.info("hogbom: Max abs in dirty image = %.6f" % numpy.fabs(dirty).max())
    absolutethresh = max(thresh, fracthresh * numpy.fabs(dirty).max())
    log.info("hogbom: Start of minor cycle")
    log.info("hogbom: This minor cycle will stop at %d iterations or peak < %s" % (niter, absolutethresh))
    
    comps = numpy.zeros(dirty.shape)
    res = numpy.array(dirty)
    pmax = psf.max()
    assert pmax > 0.0
    if window is True:
        window = 1.0
    log.info("hogbom: Max abs in dirty Image = %.6f" % numpy.fabs(res).max())
    for i in range(niter):
        if window is not None:
            mx, my = numpy.unravel_index((numpy.fabs(res * window)).argmax(), dirty.shape)
        else:
            mx, my = numpy.unravel_index((numpy.fabs(res)).argmax(), dirty.shape)
        mval = res[mx, my] * gain / pmax
        comps[mx, my] += mval
        a1o, a2o = overlapIndices(dirty, psf, mx, my)
        if i % (niter // 10) == 0:
            log.info("hogbom: Minor cycle %d, peak %s at [%d, %d]" % (i, res[mx, my], mx, my))
        res[a1o[0]:a1o[1], a1o[2]:a1o[3]] -= psf[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval
        if numpy.fabs(res).max() < absolutethresh:
            log.info("hogbom: Stopped at minor cycle %d, peak %s at [%d, %d]" % (i, res[mx, my], mx, my))
            break
    log.info("hogbom: End of minor cycles")

    return comps, res


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


