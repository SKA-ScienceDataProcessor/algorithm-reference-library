# Tim Cornwell <realtimcornwell@gmail.com>
#
# The imaging arl that have more than one data structure
#

import numpy
import pylru
from astropy import constants as const
from astropy import units as units
from astropy import wcs
from astropy.coordinates import CartesianRepresentation

from crocodile.simulate import simulate_point
from crocodile.synthesis import wcacheimg, wcachefwd, wkernaf, doimg, dopredict

from arl.define_image import create_image_from_array
from arl.define_skymodel import SkyModel
from arl.define_visibility import Visibility, combine_visibility

"""
Functions that perform imaging i.e. conversion of an Image to/from a Visibility
"""


def create_wcs_from_visibility(vt: Visibility, **kwargs):
    """Make a world coordinate system from the keyword args and Visibility
    
    :param vt:
    :type Visibility:
    :returns: WCS
    """
    print("imaging.create_wcs_from_visibility: Parsing kwargs to get definition of WCS")
    imagecentre = kwargs.get("imagecentre", vt.phasecentre)
    phasecentre = kwargs.get("phasecentre", vt.phasecentre)
    reffrequency = kwargs.get("reffrequency", numpy.max(vt.frequency)) * units.Hz
    deffaultbw = vt.frequency[0]
    if len(vt.frequency) > 1:
        deffaultbw = vt.frequency[1] - vt.frequency[0]
    channelwidth = kwargs.get("channelwidth", deffaultbw) * units.Hz
    print("imaging.create_wcs_from_visibility: Defining Image at %s, frequency %s, and bandwidth %s"
          % (imagecentre, reffrequency, channelwidth))
    
    npixel = kwargs.get("npixel", 512)
    uvmax = (numpy.abs(vt.data['uvw']).max() * reffrequency / const.c).value
    print("imaging.create_wcs_from_visibility: uvmax = %f lambda" % uvmax)
    criticalcellsize = 1.0 / (uvmax * 2.0)
    print("imaging.create_wcs_from_visibility: Critical cellsize = %f radians, %f degrees" % (
        criticalcellsize, criticalcellsize * 180.0 / numpy.pi))
    cellsize = kwargs.get("cellsize", 0.5 * criticalcellsize)
    print("imaging.create_wcs_from_visibility: Cellsize          = %f radians, %f degrees" % (cellsize,
                                                                                       cellsize * 180.0 / numpy.pi))
    if cellsize > criticalcellsize:
        print("Resetting cellsize %f radians to criticalcellsize %f radians" % (cellsize, criticalcellsize))
        cellsize = criticalcellsize
    
    npol = 4
    # Beware of python indexing order! wcs and the array have opposite ordering
    shape = [len(vt.frequency), npol, npixel, npixel]
    w = wcs.WCS(naxis=4)
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, 1.0, channelwidth.value]
    w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.value, phasecentre.dec.value, 1.0, reffrequency.value]
    w.naxis = 4
    
    w.wcs.radesys = kwargs.get('frame', 'ICRS')
    w.wcs.equinox = kwargs.get('equinox', 2000.0)
    
    return shape, reffrequency, cellsize, w, imagecentre


def invert_visibility(vt: Visibility, **kwargs):
    """Invert to make dirty Image and PSF
    
    :param vt:
    :type Visibility:
    :returns: (dirty image, psf)
    """
    print("imaging.invert_visibility: Inverting Visibility to make dirty and psf")
    shape, reffrequency, cellsize, w, imagecentre = create_wcs_from_visibility(vt, **kwargs)
    
    npixel = shape[3]
    theta = npixel * cellsize
    
    print("imaging.invert_visibility: Specified npixel=%d, cellsize = %f rad, FOV = %f rad" %
          (npixel, cellsize, theta))
    
    wstep = kwargs.get("wstep", 10000.0)
    wcachesize = int(numpy.ceil(numpy.abs(vt.data['uvw'][:, 2]).max() * reffrequency.value / (const.c.value * wstep)))
    print("imaging.invert_visibility: Making w-kernel cache of %d kernels" % wcachesize)
    wcache = pylru.FunctionCacheManager(lambda iw: wkernaf(N=256, theta=theta, w=iw * wstep, s=15, Qpx=4), 10000)
    imgfn = lambda *x: wcacheimg(*x, wstep=wstep, wcache=wcache)
    
    # Apply a phase rotation from the visibility phase centre to the image phase centre
    #    vtphaserotate = phaserotate(vt, imagecentre)
    
    d = numpy.zeros(shape)
    p = numpy.zeros(shape)
    
    pmax = 0.0
    nchan = shape[0]
    npol = shape[1]
    for channel in range(nchan):
        for pol in range(npol):
            print('imaging.invert_visibility: Inverting channel %d, polarisation %d' % (channel, pol))
            d[channel, pol, :, :], p[channel, 0, :, :], pmax = \
                doimg(theta, 1.0 / cellsize, vt.data['uvw'] *
                      (vt.frequency[channel] / const.c).value,
                      vt.data['vis'][:, channel, pol], imgfn=imgfn)
        assert pmax > 0.0, ("No data gridded for channel %d" % channel)
    
    dirty = create_image_from_array(d, w)
    psf = create_image_from_array(p, w)
    print("imaging.invert_visibility: Finished making dirty and psf")
    return dirty, psf, pmax


def predict_visibility(vt: Visibility, sm: SkyModel, **kwargs) -> Visibility:
    """Predict the visibility (in place) from a SkyModel
    
    :param vt:
    :type Visibility:
    :param sm:
    :type SkyModel:
    :returns: Visibility
    """
    vshape = vt.data['vis'].shape
    shape, reffrequency, cellsize, w, imagecentre = create_wcs_from_visibility(vt, **kwargs)
    
    vt.data['vis']=numpy.zeros(vt.data['vis'].shape)
    
    if len(sm.images):
        print("imaging.predict_visibility: Predicting Visibility from sky model images")
        
        for im in sm.images:
            wimage = im.wcs
            
            vshape = vt.data['vis'].shape
            
            ishape = sm.images[0].data.shape
            nchan = ishape[0]
            npol = ishape[1]
            npixel = ishape[3]
            
            assert ishape[0] == vshape[1], "Image and visibilityhave different number of polarisations: %d %d" % (
                ishape[1], vshape[2])
            assert ishape[0] == len(vt.frequency), "Image and visibility have different number of channels %d %d" % \
                                                   (ishape[0], len(vt.frequency))
            cellsize = abs(wimage.wcs.cdelt[0]) * numpy.pi / 180.0
            theta = npixel * cellsize
            print("imaging.predict_visibility: Image cellsize %f radians" % cellsize)
            print("imaging.predict_visibility: Field of view %f radians" % theta)
            assert (theta / numpy.sqrt(2) < 1.0), "Field of view larger than celestial sphere"
            
            wstep = kwargs.get("wstep", 10000.0)
            wcachesize = int(numpy.ceil(numpy.abs(vt.data['uvw'][:, 2]).max() * reffrequency.value / const.c.value /
                                        wstep))
            print("imaging.predict_visibility: Making w-kernel cache of %d kernels" % wcachesize)
            wcache = pylru.FunctionCacheManager(lambda iw: wkernaf(N=256, theta=theta, w=iw * wstep, s=15, Qpx=4),
                                                10000)
            predfn = lambda *x: wcachefwd(*x, wstep=wstep, wcache=wcache)
            
            for channel in range(nchan):
                uvw = vt.data['uvw'] * (vt.frequency[channel] / const.c).value
                for pol in range(npol):
                    print('imaging.predict_visibility: Predicting from image channel %d, polarisation %d' % (channel, pol))
                    puvw, dv = dopredict(theta, 1.0 / cellsize, uvw, sm.images[0].data[channel, pol, :, :],
                                         predfn=predfn)
                    vt.data['vis'][:, channel, pol] = vt.data['vis'][:, channel, pol] + dv
            print("imaging.predict_visibility: Finished predicting Visibility from sky model images")
    
    vdc = vt.phasecentre.represent_as(CartesianRepresentation)
    
    if len(sm.components):
        print("imaging.predict_visibility: Predicting Visibility from sky model components")
        
        for icomp in range(len(sm.components)):
            comp = sm.components[icomp]
            cshape = comp.flux.shape
            assert len(cshape) == 2, "Flux should be two dimensional (pol, freq)"
            nchan = cshape[0]
            npol = cshape[1]
            
            assert vshape[2] == 4, "Component %d and visibility %d have different number of polarisations" % (
                cshape[1], vshape[2])
            assert cshape[0] == len(vt.frequency), "Component %d and visibility %d have different number of channels" % \
                                                   (cshape[0], len(vt.frequency))
            
            dc = comp.direction.represent_as(CartesianRepresentation)
            print('imaging.predict_visibility: Cartesian representation of component %d = (%f, %f, %f)'
                  % (icomp, dc.x, dc.y, dc.z))
            for channel in range(nchan):
                uvw = vt.data['uvw'] * (vt.frequency[channel] / const.c).value
                uvw[:, 2] *= -1.0  # TODO: Why is this needed?
                phasor = simulate_point(uvw, dc.z, dc.y)
                for pol in range(npol):
                    print('imaging.predict_visibility: Predicting from component %d channel %d, polarisation %d' % (icomp, channel,
                                                                                                         pol))
                    vt.data['vis'][:, channel, pol] = vt.data['vis'][:, channel, pol] + \
                                                      comp.flux[channel, pol] * phasor
        print("imaging.predict_visibility: Finished predicting Visibility from sky model components")
    
    return vt


def solve_skymodel(vt: Visibility, sm: SkyModel, deconvolver, **kwargs):
    """Perform major cycles using a deconvolver. The interface of deconvolver is the same as clean.
    
    :param vt:
    :type Visibility:
    :param sm:
    :type SkyModel:
    :param deconvolver: Deconvolver to be used e.g. msclean
    :arg function:
    :returns: Visibility, SkyModel
    """
    nmajor = kwargs.get('nmajor', 5)
    print("imaging.solve_skymodel: Performing %d major cycles" % nmajor)
    
    # The model is added to each major cycle and then the visibilities are
    # calculated from the full model
    vtpred = predict_visibility(vt, sm, **kwargs)
    vtres = combine_visibility(vt, vtpred, 1.0, -1.0)
    dirty, psf, sumwt = invert_visibility(vtres, **kwargs)
    thresh = kwargs.get("threshold", 0.0)
    
    comp = sm.images[0]
    for i in range(nmajor):
        print("Start of major cycle %d" % i)
        cc, res = deconvolver(dirty, psf, **kwargs)
        comp += cc
        vtpred = predict_visibility(vt, sm, **kwargs)
        vtres = combine_visibility(vt, vtpred, 1.0, -1.0)
        dirty, psf, sumwt = invert_visibility(vtres, **kwargs)
        if numpy.abs(dirty.data).max() < 1.1 * thresh:
            print("Reached stopping threshold %.6f Jy" % thresh)
            break
        print("End of major cycle")
    print("End of major cycles")
    return vtres, sm
