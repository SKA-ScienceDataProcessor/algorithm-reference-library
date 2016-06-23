# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>)

import numpy

from .synthesis import sortw, doimg, wslicimg, wslicfwd


def overlapIndices(a1, a2,
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

    :param window: Regions where clean components are allowed. If
      True, all of the dirty Image is assumed to be allowed for
      clean components

    :param gain: The "loop gain", i.e., the fraction of the brightest
      pixel that is removed in each iteration

    :param thresh: Cleaning stops when the maximum of the absolute
      deviation of the residual is less than this value

    :param niter: Maximum number of components to make if the
      threshold `thresh` is not hit

    :returns clean SkyComponent Image, residual Image
    """

    assert 0.0 < gain < 2.0
    assert niter > 0

    comps = numpy.zeros(dirty.shape)
    res = numpy.array(dirty)
    pmax = psf.max()
    assert pmax > 0.0
    psfpeak = argmax(numpy.fabs(psf))
    if window is True:
        window = numpy.ones(dirty.shape, numpy.bool)
    for i in range(niter):
        mx, my = numpy.unravel_index((numpy.fabs(res[window])).argmax(), dirty.shape)
        mval = res[mx, my] * gain / pmax
        comps[mx, my] += mval
        a1o, a2o = overlapIndices(dirty, psf,
                                  mx - psfpeak[0],
                                  my - psfpeak[1])
        res[a1o[0]:a1o[1], a1o[2]:a1o[3]] -= psf[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval
        if numpy.fabs(res).max() < thresh:
            break
    return comps, res


def majorcycle(T2, L2,
               p, v,
               gain,
               nmajor,
               nminor,
               wstep):
    """Major cycle clean

    :param v:
    :param gain:
    :param T2: Field of view in radians
    :param L2: Observing wavelength (m)
    :param p: UVWs of visiblities (m)
    :param nmajor: Number of major cycles
    :param nminor: Number of minor cycles
    :param wstep: Step in w (pixels)

    """
    ps, vs = sortw(p, v)
    for i in range(nmajor):
        dirty, psf, _ = doimg(T2, L2, ps, vs, lambda *x: wslicimg(*x, wstep=wstep, Qpx=1))
        cc, rres = hogbom(dirty, psf, True, gain, 0,
                          nminor)
        xuv = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(cc)))
        ps, vsp = wslicfwd(xuv, T2, L2, p, wstep=wstep)
        vs = vs - vsp
    return ps, vs, cc, res
