# Tim Cornwell <realtimcornwell@gmail.com>
#
# Synthesis imaging functions
#

import numpy
import pylru
from astropy import constants as const
from astropy import units as units
from astropy import wcs
from astropy.coordinates import CartesianRepresentation

from crocodile.simulate import simulate_point
from crocodile.synthesis import wcacheimg, wcachefwd, wkernaf, doimg, dopredict

from arl.data_models import *
from arl.image_operations import create_image_from_array
from arl.parameters import get_parameter

"""
Functions that perform imaging i.e. conversion of an Image to/from a Visibility
"""


def create_wcs_from_visibility(vt: Visibility, parameters={}, level=2):
    """Make a world coordinate system from parameters and Visibility
    
    :param vt:
    :type Visibility:
    :returns: WCS
    """
    print("fourier_transformscreate_wcs_from_visibility: Parsing kwargs to get definition of WCS")
    imagecentre = get_parameter(parameters, "imagecentre", vt.phasecentre, level)
    phasecentre = get_parameter(parameters, "phasecentre", vt.phasecentre, level)
    reffrequency = get_parameter(parameters, "reffrequency", numpy.max(vt.frequency), level) * units.Hz
    deffaultbw = vt.frequency[0]
    if len(vt.frequency) > 1:
        deffaultbw = vt.frequency[1] - vt.frequency[0]
    channelwidth = get_parameter(parameters, "channelwidth", deffaultbw) * units.Hz
    print("fourier_transforms.create_wcs_from_visibility: Defining Image at %s, frequency %s, and bandwidth %s"
          % (imagecentre, reffrequency, channelwidth))
    
    npixel = get_parameter(parameters, "npixel", 512, level)
    uvmax = (numpy.abs(vt.data['uvw']).max() * reffrequency / const.c).value
    print("fourier_transforms.create_wcs_from_visibility: uvmax = %f lambda" % uvmax)
    criticalcellsize = 1.0 / (uvmax * 2.0)
    print("fourier_transforms.create_wcs_from_visibility: Critical cellsize = %f radians, %f degrees" % (
        criticalcellsize, criticalcellsize * 180.0 / numpy.pi))
    cellsize = get_parameter(parameters, "cellsize", 0.5 * criticalcellsize, level)
    print("fourier_transforms.create_wcs_from_visibility: Cellsize          = %f radians, %f degrees" % (cellsize,
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
    
    w.wcs.radesys = get_parameter(parameters, 'frame', 'ICRS')
    w.wcs.equinox = get_parameter(parameters, 'equinox', 2000.0)

    return shape, reffrequency, cellsize, w, imagecentre


def invert_visibility(vt: Visibility, parameters={}):
    """Invert to make dirty Image and PSF

    :param vt:
    :type Visibility:
    :returns: (dirty image, psf)
    """
    print("fourier_transforms.invert_visibility: Inverting Visibility to make dirty and psf")
    shape, reffrequency, cellsize, w, imagecentre = create_wcs_from_visibility(vt, parameters=parameters)
    
    npixel = shape[3]
    theta = npixel * cellsize
    
    print("fourier_transforms.invert_visibility: Specified npixel=%d, cellsize = %f rad, FOV = %f rad" %
          (npixel, cellsize, theta))
    
    # Set up the gridding kernel. We try to use a cached version
    gridding_algorithm = get_parameter(parameters, 'gridding_algorithm', 'wprojection')
    
    if gridding_algorithm == 'wprojection':
        print("fourier_transforms.invert_visibility: Gridding by w projection")
    
        wstep = get_parameter(parameters, "wstep", 10000.0)
    
        wcachesize = int(numpy.ceil(numpy.abs(vt.data['uvw'][:, 2]).max() * reffrequency.value / (const.c.value * wstep)))
        print("fourier_transforms.invert_visibility: Making w-kernel cache of %d kernels" % wcachesize)
        wcache = pylru.FunctionCacheManager(lambda iw: wkernaf(N=256, theta=theta, w=iw * wstep, s=15, Qpx=4), 10000)
        imgfn = lambda *x: wcacheimg(*x, wstep=wstep, wcache=wcache)
    else:
        raise NotImplementedError("gridding algorithm %s not supported" % gridding_algorithm)

    # Apply a phase rotation from the visibility phase centre to the image phase centre
    #    vtphaserotate = phaserotate(vt, imagecentre)
    
    d = numpy.zeros(shape)
    p = numpy.zeros(shape)

    spectral_mode = get_parameter(parameters, 'spectral_mode', 'mfs')

    if spectral_mode == 'spectral':
        pmax = 0.0
        nchan = shape[0]
        npol = shape[1]
        for channel in range(nchan):
            for pol in range(npol):
                print('fourier_transforms.invert_visibility: Inverting channel %d, polarisation %d' % (channel, pol))
                d[channel, pol, :, :], p[channel, 0, :, :], pmax = \
                    doimg(theta, 1.0 / cellsize, vt.data['uvw'] *
                          (vt.frequency[channel] / const.c).value,
                          vt.data['vis'][:, channel, pol], imgfn=imgfn)
            assert pmax > 0.0, ("No data gridded for channel %d" % channel)
    elif spectral_mode == 'mfs':
        pmax = 0.0
        nchan = shape[0]
        npol = shape[1]
        for channel in range(nchan):
            for pol in range(npol):
                print('fourier_transforms.invert_visibility: Inverting channel %d, polarisation %d' % (channel, pol))
                d[0, pol, :, :], p[0, 0, :, :], pmax = \
                    doimg(theta, 1.0 / cellsize, vt.data['uvw'] * (vt.frequency[channel] / const.c).value,
                          vt.data['vis'][:, channel, pol], imgfn=imgfn)
            assert pmax > 0.0, ("No data gridded for channel %d" % channel)
    else:
        raise NotImplementedError("mode %s not supported" % spectral_mode)


    dirty = create_image_from_array(d, w)
    psf = create_image_from_array(p, w)
    print("fourier_transforms.invert_visibility: Finished making dirty and psf")


    return dirty, psf, pmax


def predict_visibility(vt: Visibility, sm: SkyModel, parameters={}) -> Visibility:
    """Predict the visibility (in place) from a SkyModel

    :param vt:
    :type Visibility:
    :param sm:
    :type SkyModel:
    :returns: Visibility
    """
    vshape = vt.data['vis'].shape
    shape, reffrequency, cellsize, w, imagecentre = create_wcs_from_visibility(vt, parameters=parameters, level=2)
    
    vt.data['vis'] = numpy.zeros(vt.data['vis'].shape)
    
    if len(sm.images):
        print("fourier_transforms.predict_visibility: Predicting Visibility from sky model images")
        
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
            print("fourier_transforms.predict_visibility: Image cellsize %f radians" % cellsize)
            print("fourier_transforms.predict_visibility: Field of view %f radians" % theta)
            assert (theta / numpy.sqrt(2) < 1.0), "Field of view larger than celestial sphere"
            
            wstep = get_parameter(parameters, "wstep", 10000.0)
            wcachesize = int(numpy.ceil(numpy.abs(vt.data['uvw'][:, 2]).max() * reffrequency.value / const.c.value /
                                        wstep))
            print("fourier_transforms.predict_visibility: Making w-kernel cache of %d kernels" % wcachesize)
            wcache = pylru.FunctionCacheManager(lambda iw: wkernaf(N=256, theta=theta, w=iw * wstep, s=15, Qpx=4),
                                                10000)
            predfn = lambda *x: wcachefwd(*x, wstep=wstep, wcache=wcache)
            
            for channel in range(nchan):
                uvw = vt.data['uvw'] * (vt.frequency[channel] / const.c).value
                for pol in range(npol):
                    print('fourier_transforms.predict_visibility: Predicting from image channel %d, polarisation %d' % (
                    channel, pol))
                    puvw, dv = dopredict(theta, 1.0 / cellsize, uvw, sm.images[0].data[channel, pol, :, :],
                                         predfn=predfn)
                    vt.data['vis'][:, channel, pol] = vt.data['vis'][:, channel, pol] + dv
            print("fourier_transforms.predict_visibility: Finished predicting Visibility from sky model images")
    
    vdc = vt.phasecentre.represent_as(CartesianRepresentation)
    
    if len(sm.components):
        print("fourier_transforms.predict_visibility: Predicting Visibility from sky model components")
        
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
            print('fourier_transforms.predict_visibility: Cartesian representation of component %d = (%f, %f, %f)'
                  % (icomp, dc.x, dc.y, dc.z))
            for channel in range(nchan):
                uvw = vt.data['uvw'] * (vt.frequency[channel] / const.c).value
                uvw[:, 2] *= -1.0  # TODO: Why is this needed?
                phasor = simulate_point(uvw, dc.z, dc.y)
                for pol in range(npol):
                    print(
                        'fourier_transforms.predict_visibility: Predicting from component %d channel %d, polarisation %d' % (
                        icomp, channel,
                        pol))
                    vt.data['vis'][:, channel, pol] = vt.data['vis'][:, channel, pol] + \
                                                      comp.flux[channel, pol] * phasor
        print("fourier_transforms.predict_visibility: Finished predicting Visibility from sky model components")
    
    return vt


def weight_visibility(vt, im, parameters={}):
    """ Reweight the visibility data in place a selected algorithm

    :param vt:
    :type Visibility:
    :param im:
    :type Image:
    :param parameters:
    :returns: Configuration
    """
    print("visibility_operations.weight_visibility: not yet implemented")
    return vt

