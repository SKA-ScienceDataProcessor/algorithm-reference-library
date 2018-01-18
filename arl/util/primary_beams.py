"""
Functions to create primary beam modelsw
"""

import numpy
import warnings

from astropy import constants as const
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs import FITSFixedWarning

from arl.image.operations import create_empty_image_like, fft_image

import logging

log = logging.getLogger(__name__)


def ft_disk(r):
    from scipy.special import jn  # pylint: disable=no-name-in-module
    result = numpy.zeros_like(r)
    result[r > 0] = 2.0 * jn(1, r[r > 0]) / r[r > 0]
    rsmall = 1e-9
    result[r == 0] = 2.0 * jn(1, rsmall) / rsmall
    return result


def create_pb_vla(model, pointingcentre=None):
    """
    Make an image like model and fill it with an analytical model of the primary beam
    :param model:
    :return:
    """
    beam = create_empty_image_like(model)
    
    nchan, npol, ny, nx = model.shape
    
    if pointingcentre is not None:
        cx, cy = skycoord_to_pixel(pointingcentre, model.wcs, 0, 'wcs')
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FITSFixedWarning)
            cx, cy = beam.wcs.sub(2).wcs.crpix[0] - 1, beam.wcs.sub(2).wcs.crpix[1] - 1
    
    for chan in range(nchan):
        
        # The frequency axis is the second to last in the beam
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FITSFixedWarning)
            frequency = model.wcs.sub(['spectral']).wcs_pix2world([chan], 0)[0]
        wavelength = const.c.to('m/s').value / frequency

        d2r = numpy.pi / 180.0
        scale = d2r * numpy.abs(beam.wcs.sub(2).wcs.cdelt[0])
        xx, yy = numpy.meshgrid(scale * (range(nx) - cx), scale * (range(ny) - cy))
        # Radius of each cell in radians
        rr = numpy.sqrt(xx ** 2 + yy ** 2)
        
        for pol in range(npol):
            reflector = ft_disk(rr * numpy.pi * 25.0 / wavelength)
            # blockage = ft_disk(rr * numpy.pi * 1.67 / wavelength)
            beam.data[chan, pol, ...] = reflector  # - blockage
    
    beam.data *= beam.data
    return beam


def create_illum_vla(model):
    def disk(a, xx, yy, radius):
        disk = numpy.zeros_like(a)
        nx, ny = a.shape
        rr = numpy.sqrt(xx ** 2 + yy ** 2)
        for y in range(ny):
            for x in range(nx):
                if rr[x, y] < radius / 2:
                    disk[x, y] = 1.0
        return disk
    
    nchan, npol, ny, nx = model.shape
    
    # The beam is assumed to just scale with frequency.
    
    beam = create_empty_image_like(model)
    illum = fft_image(beam)
    
    for chan in range(nchan):
        
        # The frequency axis is the second to last in the beam
        frequency = model.wcs.sub(['spectral']).wcs_pix2world([chan], 0)[0]
        wavelength = const.c.to('m/s').value / frequency
        scaleuv = numpy.abs(illum.wcs.sub(2).wcs.cdelt[0]) * wavelength
        
        xx, yy = numpy.meshgrid(scaleuv * (range(nx) - illum.wcs.sub(2).wcs.crpix[0] - 1),
                                scaleuv * (range(ny) - illum.wcs.sub(2).wcs.crpix[1] - 1))
        
        for pol in range(npol):
            reflector = disk(illum.data[chan, pol, ...], xx, yy, 25.0)
            blockage = disk(illum.data[chan, pol, ...], xx, yy, 1.67)
            illum.data[chan, pol, ...] = reflector - blockage
    
    return illum


def create_illum_vla_numerical(model):
    def disk(a, xx, yy, radius):
        disk = numpy.zeros_like(a)
        nx, ny = a.shape
        rr = numpy.sqrt(xx ** 2 + yy ** 2)
        for y in range(ny):
            for x in range(nx):
                if rr[x, y] < radius / 2:
                    disk[x, y] = 1.0
        return disk
    
    nchan, npol, ny, nx = model.shape
    
    # The beam is assumed to just scale with frequency.
    
    beam = create_empty_image_like(model)
    illum = fft_image(beam)
    
    for chan in range(nchan):
        
        # The frequency axis is the second to last in the beam
        frequency = model.wcs.sub(['spectral']).wcs_pix2world([chan], 0)[0]
        wavelength = const.c.to('m/s').value / frequency
        scaleuv = numpy.abs(illum.wcs.sub(2).wcs.cdelt[0]) * wavelength
        
        xx, yy = numpy.meshgrid(scaleuv * (range(nx) - illum.wcs.sub(2).wcs.crpix[0] - 1),
                                scaleuv * (range(ny) - illum.wcs.sub(2).wcs.crpix[1] - 1))
        
        for pol in range(npol):
            reflector = disk(illum.data[chan, pol, ...], xx, yy, 25.0)
            blockage = disk(illum.data[chan, pol, ...], xx, yy, 1.67)
            illum.data[chan, pol, ...] = reflector - blockage
    
    return illum


def create_efield_vla_numerical(model):
    efield = fft_image(create_illum_vla(model), model)
    return efield


def create_pb_vla_numerical(model):
    pb = create_efield_vla_numerical(model)
    pb.data *= pb.data
    pb.data = numpy.real(pb.data).astype('float')
    return pb
