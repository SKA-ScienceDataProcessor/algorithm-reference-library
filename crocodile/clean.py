# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>)

import numpy

from synthesis import sortw, doimg, wslicimg, wslicfwd

def overlapIndices(a1, a2, 
                   shiftx, shifty):
    if shiftx >=0:
        a1xbeg=shiftx
        a2xbeg=0
        a1xend=a1.shape[0]
        a2xend=a1.shape[0]-shiftx
    else:
        a1xbeg=0
        a2xbeg=-shiftx
        a1xend=a1.shape[0]+shiftx
        a2xend=a1.shape[0]

    if shifty >=0:
        a1ybeg=shifty
        a2ybeg=0
        a1yend=a1.shape[1]
        a2yend=a1.shape[1]-shifty
    else:
        a1ybeg=0
        a2ybeg=-shifty
        a1yend=a1.shape[1]+shifty
        a2yend=a1.shape[1]

    return (a1xbeg, a1xend, a1ybeg, a1yend), (a2xbeg, a2xend, a2ybeg, a2yend)

def argmax(a):
    return numpy.unravel_index(a.argmax(), a.shape)

def hogbom(dirty,
           psf,
           window,
           gain,
           thresh,
           niter):
    """
    Hogbom CLEAN (1974A&AS...15..417H)

    :param dirty: The dirty image, i.e., the image to be deconvolved

    :param psf: The point spread-function

    :param window: Regions where clean components are allowed. If
      True, thank all of the dirty image is assumed to be allowed for
      clean components

    :param gain: The "loop gain", i.e., the fraction of the brightest
      pixel that is removed in each iteration

    :param thresh: Cleaning stops when the maximum of the absolute
      deviation of the residual is less than this value

    :param niter: Maximum number of components to make if the
      threshold `thresh` is not hit
    """
    comps=numpy.zeros(dirty.shape)
    res=numpy.array(dirty)
    pmax=psf.max()
    psfpeak=argmax(numpy.fabs(psf))
    if window is True:
        window=numpy.ones(dirty.shape,
                          numpy.bool)
    for i in range(niter):
        mx, my=numpy.unravel_index((numpy.fabs(res[window])).argmax(), dirty.shape)
        mval=res[mx, my]*gain/pmax
        comps[mx, my]+=mval
        a1o, a2o=overlapIndices(dirty, psf,
                                mx-psfpeak[0],
                                my-psfpeak[1])
        res[a1o[0]:a1o[1],a1o[2]:a1o[3]]-=psf[a2o[0]:a2o[1],a2o[2]:a2o[3]]*mval
        if numpy.fabs(res).max() < thresh:
            break
    return comps, res
        

def majorcycle(T2, L2,
               p, v,
               gain,
               nmajor,
               nminor,
               wstep):
    "Major cycle clean"
    ps, vs = sortw(p, v)
    for i in range(nmajor):
        dirty,psf=doimg(T2, L2, ps, vs, lambda *x: wslicimg(*x, wstep=wstep, Qpx=1))
        cc,rres=hogbom(dirty, psf, True, gain, 0,
                       nminor)
        xuv=numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(cc)))
        ps, vsp=wslicfwd(xuv, T2, L2, p, wstep=wstep)
        vs=vs-vsp
    return ps, vs

    
