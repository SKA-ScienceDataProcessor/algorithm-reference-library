# Tim Cornwell <realtimcornwell@gmail.com>)

import numpy

from synthesis import sortw, doimg, wslicimg, wslicfwd

from clean import overlapIndices, argmax

def majorcycle(T2, L2,
               p, v,
               gain,
               nmajor,
               nminor,
               wstep,
               scales):
    "Major cycle for msclean"
    ps, vs = sortw(p, v)
    for i in range(nmajor):
        dirty,psf=doimg(T2, L2, ps, vs, lambda *x: wslicimg(*x, wstep=wstep, Qpx=1))
        cc,rres=msclean(dirty, psf, True, gain, 0,
                       nminor, scales)
        xuv=numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(cc)))
        ps, vsp=wslicfwd(xuv, T2, L2, p, wstep=wstep)
        vs=vs-vsp
    return ps, vs, cc, rres

def msclean(dirty,
           psf,
           window,
           gain,
           thresh,
           niter,
           scales):
    """
    Multiscale CLEAN minor cycle (IEEE Journal of Selected Topics in Sig Proc, 2008 vol. 2 pp. 793-801)

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
    threshold "thresh" is not hit
    
    :param scales: Scales (in pixels width) to be used    
    
    """
    comps=numpy.zeros(dirty.shape)

    pmax=psf.max()
    psfpeak=argmax(numpy.fabs(psf))
    print "Peak of PSF = %s at %s" % (pmax, psfpeak)
    dmax=dirty.max()
    dpeak=argmax(dirty)
    print "Peak of Dirty = %s at %s" % (dmax, dpeak)
    lpsf=psf/pmax
    ldirty=dirty/pmax
    
    """ Create the scale images and form all the various products we need
    """
    scaleshape=[ldirty.shape[0], ldirty.shape[1], len(scales)]
    scalescaleshape=[ldirty.shape[0], ldirty.shape[1], len(scales), len(scales)]
    scalestack=createscalestack(scaleshape,scales)
    
    couplingMatrix=numpy.zeros([len(scales),len(scales)])
    psfscalestack=convolvescalestack(scalestack, numpy.array(lpsf))
    resscalestack=convolvescalestack(scalestack, numpy.array(ldirty))
    """ Evaluate the coupling matrix
    """
    psfscalescalestack=numpy.zeros(scalescaleshape)
    for iscale in numpy.arange(len(scales)):
        psfscalescalestack[:,:,:,iscale]=convolvescalestack(scalestack,psfscalestack[:,:,iscale])
        psfscalescalestack[:,:,iscale,:]=psfscalescalestack[:,:,:,iscale]
    for iscale in numpy.arange(len(scales)):
        for iscale1 in numpy.arange(len(scales)):
            couplingMatrix[iscale,iscale1]=numpy.max(psfscalescalestack[:,:,iscale,iscale1])
    print "Coupling matrix = %s" % couplingMatrix
    
    """ The window is scale dependent - we form it by smoothing and thresholding
    the input window
    """
    if window is True:
        windowstack=numpy.ones(scalestack.shape, numpy.bool)
#    windowstack=convolvescalestack(scalestack, window)>0.9
    window=numpy.ones(scalestack.shape, numpy.bool)
    
    """ The minor cycle
    """
    for i in range(niter):
        """Find peak over all smoothed images
        """
        resmax=0.0
        mscale=0
        for iscale in numpy.arange(len(scales)):
            mx, my=numpy.unravel_index(numpy.fabs((resscalestack)[:,:,iscale]).argmax(), dirty.shape)
            thismax=resscalestack[mx,my,iscale]
            if thismax>resmax:
                resmax=thismax
                mscale=iscale
                mx, my=numpy.unravel_index((numpy.fabs(resscalestack[:,:,iscale])).argmax(), dirty.shape)
                
        """ Find the values to subtract, accounting for the coupling matrix
        """
        mval=numpy.zeros(len(scales))
        for iscale in numpy.arange(len(scales)):
            mval[iscale]=resscalestack[mx, my, iscale]*gain/couplingMatrix[iscale,iscale]
        print "Iteration %d, peak %s at [%d, %d, %d]" % (i, mval, mx, my, mscale)
        
        """ Subtract from the residuals and add to the model
        """
        a1o, a2o=overlapIndices(dirty, psf,
                                mx-psfpeak[0],
                                my-psfpeak[1])
        for iscale in numpy.arange(len(scales)):
            resscalestack[a1o[0]:a1o[1],a1o[2]:a1o[3],iscale]-=psfscalestack[a2o[0]:a2o[1],a2o[2]:a2o[3],iscale]*mval[iscale]
            comps[a1o[0]:a1o[1],a1o[2]:a1o[3]]+=scalestack[a2o[0]:a2o[1],a2o[2]:a2o[3],iscale]*mval[iscale]
        if numpy.fabs(resscalestack[:,:,0]).max() < thresh:
            break
    return comps, resscalestack[:,:,0]
    
def createscalestack(scaleshape,scales):
    """ Create a cube consisting of the scales
    """
    basis=numpy.zeros(scaleshape)
    nx=scaleshape[0]
    ny=scaleshape[1]
    xcen=nx/2
    ycen=ny/2
    for iscale in numpy.arange(0,len(scales)):
        scale=scales[iscale]
        rscale2=1.0/(float(scale)/2.0)**2
        x=range(xcen-scale/2-1,xcen+scale/2+1)
        fx=numpy.array(x, 'float')-float(xcen)
        for y in range(ycen-scale/2-1,ycen+scale/2+1):
            fy=float(y-ycen)
            r2=fx*fx+fy*fy
            basis[x,y,iscale]=(1.0-r2*rscale2)
        basis[basis<0.0]=0.0
    return basis
         
def convolvescalestack(scalestack, img):
    """Convolve img by the specified scalestack, returning the resulting stack
    """
    convolved=numpy.zeros(scalestack.shape)
    ximg=numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(img)))

    nscales=scalestack.shape[2]
    for iscale in range(nscales):
        xscale=numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(scalestack[:,:,iscale])))
        xmult=ximg*xscale
        convolved[:,:,iscale]=numpy.real(numpy.fft.ifftshift(numpy.fft.ifft2(numpy.fft.ifftshift(xmult))))
    return convolved    
