# Tim Cornwell <realtimcornwell@gmail.com>)

import numpy

from crocodile.synthesis import sortw, doimg, wslicimg, wslicfwd

from crocodile.clean import overlapIndices, argmax

def majorcycle(T2, L2,
               p, v,
               gain,
               nmajor,
               nminor,
               wstep,
               scales,
               thresh=0.0):
    """Major cycle for msclean

    :param T2: Field of view in radians
    :param L2: Observing wavelength (m)
    :param p: UVWs of visiblities (m)
    :param v: Values of visibilities
    :param nmajor: Number of major cycles
    :param nminor: Number of minor cycles
    :param wstep: Step in w (pixels)
    :param wscales: Array of scales in pixels
    :print thresh: Stopping threshold (for scale=0)
    """
    ps, vs = sortw(p, v)
    for i in range(nmajor):
        dirty,psf=doimg(T2, L2, ps, vs, lambda *x: wslicimg(*x, wstep=wstep))
        cc,rres=msclean(dirty, psf, True, gain, thresh, nminor, scales)
        norm=float(cc.shape[0])*float(cc.shape[1])
        guv=numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.fftshift(cc)))*norm
        ps, vsp=wslicfwd(guv, T2, L2, p, wstep=wstep)
        vs=vs-vsp
        rres,psf=doimg(T2, L2, ps, vs, lambda *x: wslicimg(*x, wstep=wstep))
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
    :returns clean component image, residual image
    """
    assert gain > 0.0 and gain < 2.0
    assert niter > 0
    assert len(scales) > 0

    comps=numpy.zeros(dirty.shape)

    pmax=psf.max()
    assert pmax> 0.0

    psfpeak=argmax(numpy.fabs(psf))
    print ("Peak of PSF = %s at %s" % (pmax, psfpeak))
    dmax=dirty.max()
    dpeak=argmax(dirty)
    print ("Peak of Dirty = %s at %s" % (dmax, dpeak))
    lpsf=psf/pmax
    ldirty=dirty/pmax

    # Create the scale images and form all the various products we need. We
    # use a third dimension to hold the scale-related images. scalestack is a 3D
    # cube holding the different scale images. convolvestack will take a 2D image
    # and add a third dimension holding the scale-convolved versions.

    scaleshape=[ldirty.shape[0], ldirty.shape[1], len(scales)]
    scalescaleshape=[ldirty.shape[0], ldirty.shape[1], len(scales), len(scales)]
    scalestack=createscalestack(scaleshape,scales)

    couplingMatrix=numpy.zeros([len(scales),len(scales)])
    psfscalestack=convolvescalestack(scalestack, numpy.array(lpsf))
    resscalestack=convolvescalestack(scalestack, numpy.array(ldirty))
    # Evaluate the coupling matrix between the various scale sizes.
    psfscalescalestack=numpy.zeros(scalescaleshape)
    for iscale in numpy.arange(len(scales)):
        psfscalescalestack[:,:,:,iscale]=convolvescalestack(scalestack,psfscalestack[:,:,iscale])
        psfscalescalestack[:,:,iscale,:]=psfscalescalestack[:,:,:,iscale]
    for iscale in numpy.arange(len(scales)):
        for iscale1 in numpy.arange(len(scales)):
            couplingMatrix[iscale,iscale1]=numpy.max(psfscalescalestack[:,:,iscale,iscale1])
    print ("Coupling matrix = %s" % couplingMatrix)

    # The window is scale dependent - we form it by smoothing and thresholding
    # the input window. This prevents components being placed too close to the
    # edge of the image.

    if window is True:
        windowstack=numpy.ones(scalestack.shape, numpy.bool)
#    windowstack=convolvescalestack(scalestack, window)>0.9
    window=numpy.ones(scalestack.shape, numpy.bool)

    """ The minor cycle
    """
    for i in range(niter):
        # Find peak over all smoothed images
        resmax=0.0
        mscale=0
        for iscale in numpy.arange(len(scales)):
            mx, my=numpy.unravel_index(numpy.fabs((resscalestack)[:,:,iscale]).argmax(), dirty.shape)
            thismax=resscalestack[mx,my,iscale]/couplingMatrix[mscale,iscale]
            if thismax>resmax:
                resmax=thismax
                mscale=iscale
                mx, my=numpy.unravel_index((numpy.fabs(resscalestack[:,:,iscale])).argmax(), dirty.shape)

        # Find the values to subtract, accounting for the coupling matrix
        mval=numpy.zeros(len(scales))
        for iscale in numpy.arange(len(scales)):
            mval[iscale]=resscalestack[mx, my, iscale]*gain/couplingMatrix[iscale,iscale]
        print ("Iteration %d, peak %s at [%d, %d, %d]" % (i, mval, mx, my, mscale))

        #  Update the cached residuals and add to the cached model.
        a1o, a2o=overlapIndices(dirty, psf,
                                mx-psfpeak[0],
                                my-psfpeak[1])
        for iscale in numpy.arange(len(scales)):
            resscalestack[a1o[0]:a1o[1],a1o[2]:a1o[3],iscale]-= psfscalescalestack[a2o[0]:a2o[1],a2o[2]:a2o[3],mscale,iscale]* \
                mval[iscale]
            comps[a1o[0]:a1o[1],a1o[2]:a1o[3]]+=scalestack[a2o[0]:a2o[1],a2o[2]:a2o[3],iscale]*mval[iscale]
        if numpy.fabs(resscalestack[:,:,0]).max() < thresh:
            break
    return comps, pmax*resscalestack[:,:,0]

def createscalestack(scaleshape,scales):
    """ Create a cube consisting of the scales

    :param scaleshape: desired shape of stack
    :param scales: scales (in pixels)
    """
    assert scaleshape[2] == len(scales)

    basis=numpy.zeros(scaleshape)
    nx=scaleshape[0]
    ny=scaleshape[1]
    xcen=int(numpy.ceil(float(nx)/2.0))
    ycen=int(numpy.ceil(float(ny)/2.0))
    for iscale in numpy.arange(0,len(scales)):
        halfscale=int(numpy.ceil(scales[iscale]/2.0))
        rscale2=1.0/(float(scales[iscale])/2.0)**2
        x=range(xcen-halfscale-1,xcen+halfscale+1)
        fx=numpy.array(x, 'float')-float(xcen)
        for y in range(ycen-halfscale-1,ycen+halfscale+1):
            fy=float(y-ycen)
            r2=fx*fx+fy*fy
            basis[x,y,iscale]=(1.0-r2*rscale2)
        basis[basis<0.0]=0.0
    return basis

def convolvescalestack(scalestack, img):
    """Convolve img by the specified scalestack, returning the resulting stack

    :param scalestack: stack containing the scales
    :param img: image to be convolved
    """

    convolved=numpy.zeros(scalestack.shape)
    ximg=numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(img)))

    nscales=scalestack.shape[2]
    for iscale in range(nscales):
        xscale=numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(scalestack[:,:,iscale])))
        xmult=ximg*xscale
        convolved[:,:,iscale]=numpy.real(numpy.fft.ifftshift(numpy.fft.ifft2(numpy.fft.ifftshift(xmult))))
    return convolved
