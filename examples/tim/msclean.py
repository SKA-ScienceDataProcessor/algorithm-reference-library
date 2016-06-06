# Tim Cornwell <realtimcornwell@gmail.com>)

import numpy

from crocodile.synthesis import sortw, doimg, dopredict, wslicimg, wslicfwd

from crocodile.clean import overlapIndices, argmax

import matplotlib.pyplot as plt

def majorcycle(T2, L2,
               p, v,
               gain,
               nmajor,
               nminor,
               wstep,
               scales,
               thresh=0.0,
               fracthresh=0.1):
    """Major cycle for MultiScale Clean

    :param T2: Field of view in radians
    :param L2: Maximum uv (for setting image pixel sizes)
    :param p: UVWs of visibilities (m)
    :param v: Values of visibilities
    :param nmajor: Number of major cycles
    :param nminor: Number of minor cycles
    :param wstep: Step in w (pixels)
    :param wscales: Array of scales in pixels
    :print thresh: Stopping threshold (for scale=0)
    :print fracthresh: Minor Cycle stopping threshold (for scale=0) fraction of peak
    """

    # The model is added to each major cycle and then the visibilities are
    # calculated from the full model
    ps, vso = sortw(p, v)
    dirty,psf=doimg(T2, L2, ps, vso, lambda *x: wslicimg(*x, wstep=wstep))
    comps=0.0*dirty.copy()
    for i in range(nmajor):
        print("Start of major cycle %d" % (i))
        cc, res=msclean(dirty, psf, True, gain, thresh, nminor, scales, fracthresh)
        plt.clf()
        plt.imshow(res,cmap='rainbow', origin='lower')
        plt.colorbar()
        plt.show()
        comps+=cc
        # dopredict resorts the data
        pss, vsp=dopredict(T2, L2, ps, comps, lambda *x: wslicfwd(*x, wstep=wstep))
        vsr=vso-vsp
        dirty, psf=doimg(T2, L2, ps, vsr, lambda *x: wslicimg(*x, wstep=wstep))
        print("End of major cycle")
    print("End of major cycles")
    return ps, vsp, comps, dirty

def msclean(dirty,
           psf,
           window,
           gain,
           thresh,
           niter,
           scales,
           fracthresh):
    """
    Multiscale CLEAN minor cycle (IEEE Journal of Selected Topics in Sig Proc, 2008 vol. 2 pp. 793-801)

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
    scalestack=createscalestack(scaleshape,scales, norm=True)

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
    print ("Coupling matrix =\n %s" % couplingMatrix)

    # The window is scale dependent - we form it by smoothing and thresholding
    # the input window. This prevents components being placed too close to the
    # edge of the image.

    if window is True:
        windowstack=numpy.ones(scalestack.shape, numpy.bool)
#    windowstack=convolvescalestack(scalestack, window)>0.9
    window=numpy.ones(scalestack.shape, numpy.bool)

    """ The minor cycle
    """
    print("Max abs in dirty image = %.6f" % numpy.fabs(resscalestack[:,:,0]).max())
    absolutethresh=max(thresh,fracthresh*numpy.fabs(resscalestack[:,:,0]).max())
    print("Start of minor cycle")
    print("This minor cycle will stop at %d iterations or peak < %s" % (niter, absolutethresh))

    for i in range(niter):
        # Find peak over all smoothed images
        mx,my,mscale=findabsmaxstack(resscalestack, window, couplingMatrix)
        if mx == None or my == None or mscale == None:
            print("Error in finding peak")
            break

        # Find the values to subtract, accounting for the coupling matrix
        mval=numpy.zeros(len(scales))
        mval[mscale]=resscalestack[mx, my, mscale]/couplingMatrix[mscale,mscale]
        print ("Minor cycle %d, peak %s at [%d, %d, %d]" % \
            (i, resscalestack[mx, my, :], mx, my, mscale))
        if numpy.fabs(mval[mscale]) < absolutethresh:
            print("Absolute value of peak %.6f is below stopping threshold %.6f" \
                % (numpy.fabs(resscalestack[mx,my,mscale]), absolutethresh))
            break

        #  Update the cached residuals and add to the cached model.
        a1o, a2o=overlapIndices(dirty, psf, mx-psfpeak[0], my-psfpeak[1])
        if numpy.abs(mval[mscale])>0:
            # Cross subtract from other scales
            for iscale in range(len(scales)):
                resscalestack[a1o[0]:a1o[1],a1o[2]:a1o[3],iscale]-= \
                    psfscalescalestack[a2o[0]:a2o[1],a2o[2]:a2o[3],iscale,mscale] * \
                    gain * mval[mscale]
            comps[a1o[0]:a1o[1],a1o[2]:a1o[3]] += \
                scalestack[a2o[0]:a2o[1],a2o[2]:a2o[3],mscale] * \
                gain * mval[mscale]
        else:
            break
    print("End of minor cycles")
    return comps, pmax*resscalestack[:,:,0]

def createscalestack(scaleshape,scales, norm=False):
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
        if scales[iscale]>0.0:
            rscale2=1.0/(float(scales[iscale])/2.0)**2
            x=range(xcen-halfscale-1,xcen+halfscale+1)
            fx=numpy.array(x, 'float')-float(xcen)
            for y in range(ycen-halfscale-1,ycen+halfscale+1):
                fy=float(y-ycen)
                r2=fx*fx+fy*fy
                basis[x,y,iscale]=(1.0-r2*rscale2)
            basis[basis<0.0]=0.0
            if norm:
                basis[:,:,iscale]/=numpy.sum(basis[:,:,iscale])
        else:
            basis[xcen,ycen,iscale]=1.0
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

def findabsmaxstack(stack, window, couplingMatrix):
    """Find the location and value of the absolute maximum in this stack
    """
    pabsmax=0.0
    pscale=None
    px=None
    py=None
    pshape=[stack.shape[0],stack.shape[1]]
    for iscale in range(stack.shape[2]):
        mx, my=numpy.unravel_index(numpy.fabs((stack)[:,:,iscale]).argmax(), pshape)
        thisabsmax=stack[mx,my,iscale]/couplingMatrix[iscale,iscale]
        if abs(thisabsmax)>abs(pabsmax):
            px=mx
            py=my
            pscale=iscale
            pabsmax=stack[px,py,pscale]/couplingMatrix[iscale,iscale]
    return px, py, pscale
