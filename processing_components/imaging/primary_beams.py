"""
Functions to create primary beam modelsw
"""

import collections
import logging

import numpy
from astropy import constants as const

from data_models.memory_data_models import Image
from data_models.parameters import arl_path
from processing_components.image.operations import create_empty_image_like
from ..image.operations import import_image_from_fits, create_image_from_array, \
    reproject_image

log = logging.getLogger(__name__)


def ft_disk(r):
    from scipy.special import jn  # pylint: disable=no-name-in-module
    result = numpy.zeros_like(r)
    result[r > 0] = 2.0 * jn(1, r[r > 0]) / r[r > 0]
    rsmall = 1e-9
    result[r == 0] = 2.0 * jn(1, rsmall) / rsmall
    return result


def create_pb(model, telescope='MID', pointingcentre=None):
    """
    Make an image like model and fill it with an analytical model of the primary beam
    :param model: Template image
    :param telescope: 'VLA' or 'ASKAP'
    :return: Primary beam image
    """
    if telescope[0:3] == 'MID':
        return create_pb_generic(model, pointingcentre=pointingcentre, diameter=15.0, blockage=0.0)
    elif telescope[0:3] == 'LOW':
        return create_low_test_beam(model)
    elif telescope[0:3] == 'VLA':
        return create_pb_generic(model, pointingcentre=pointingcentre, diameter=25.0, blockage=1.8)
    elif telescope[0:5] == 'ASKAP':
        return create_pb_generic(model, pointingcentre=pointingcentre, diameter=12.0, blockage=1.0)
    else:
        raise NotImplementedError('Telescope %s has no primary beam model' % telescope)


def mosaic_pb(model, telescope, pointingcentres):
    """ Create a mosaic primary beam by adding primary beams for a set of pointing centres
    
    Note that the addition is root sum of squares
    
    :param model:  Template image
    :param telescope:
    :param pointingcentres:  list of pointing centres
    :return:
    """
    assert isinstance(pointingcentres, collections.Iterable), "Need a list of pointing centres"
    sumpb = create_empty_image_like(model)
    for pc in pointingcentres:
        pb = create_pb(model, telescope, pointingcentre=pc)
        sumpb.data += pb.data ** 2
    sumpb.data = numpy.sqrt(sumpb.data)
    return sumpb


def create_pb_generic(model, pointingcentre=None, diameter=25.0, blockage=1.8):
    """
    Make an image like model and fill it with an analytical model of the primary beam
    :param model:
    :return:
    """
    beam = create_empty_image_like(model)
    
    nchan, npol, ny, nx = model.shape
    
    if pointingcentre is not None:
        cx, cy = pointingcentre.to_pixel(model.wcs, origin=0)
    else:
        cx, cy = beam.wcs.sub(2).wcs.crpix[0] - 1, beam.wcs.sub(2).wcs.crpix[1] - 1
    
    for chan in range(nchan):
        
        # The frequency axis is the second to last in the beam
        frequency = model.wcs.sub(['spectral']).wcs_pix2world([chan], 0)[0]
        wavelength = const.c.to('m s^-1').value / frequency
        
        d2r = numpy.pi / 180.0
        scale = d2r * numpy.abs(beam.wcs.sub(2).wcs.cdelt[0])
        xx, yy = numpy.meshgrid(scale * (range(nx) - cx), scale * (range(ny) - cy))
        # Radius of each cell in radians
        rr = numpy.sqrt(xx ** 2 + yy ** 2)
        
        blockage_factor = (blockage / diameter) ** 2
        
        for pol in range(npol):
            reflector = ft_disk(rr * numpy.pi * diameter / wavelength)
            blockage = ft_disk(rr * numpy.pi * blockage / wavelength)
            beam.data[chan, pol, ...] = reflector - blockage_factor * blockage
    
    beam.data *= beam.data
    return beam


def create_low_test_beam(model: Image) -> Image:
    """Create a test power beam for LOW using an image from OSKAR

    :param model: Template image
    :return: Image
    """
    
    beam = import_image_from_fits(arl_path('data/models/SKA1_LOW_beam.fits'))
    
    # Scale the image cellsize to account for the different in frequencies. Eventually we will want to
    # use a frequency cube
    log.info("create_low_test_beam: primary beam is defined at %.3f MHz" % (beam.wcs.wcs.crval[2] * 1e-6))
    
    nchan, npol, ny, nx = model.shape
    
    # We need to interpolate each frequency channel separately. The beam is assumed to just scale with
    # frequency.
    
    reprojected_beam = create_empty_image_like(model)
    
    for chan in range(nchan):
        
        model2dwcs = model.wcs.sub(2).deepcopy()
        model2dshape = [model.shape[2], model.shape[3]]
        beam2dwcs = beam.wcs.sub(2).deepcopy()
        
        # The frequency axis is the second to last in the beam
        frequency = model.wcs.sub(['spectral']).wcs_pix2world([chan], 0)[0]
        fscale = beam.wcs.wcs.crval[2] / frequency
        
        beam2dwcs.wcs.cdelt = fscale * beam.wcs.sub(2).wcs.cdelt
        beam2dwcs.wcs.crpix = beam.wcs.sub(2).wcs.crpix
        beam2dwcs.wcs.crval = model.wcs.sub(2).wcs.crval
        beam2dwcs.wcs.ctype = model.wcs.sub(2).wcs.ctype
        model2dwcs.wcs.crpix = [model.shape[2] // 2 + 1, model.shape[3] // 2 + 1]
        
        beam2d = create_image_from_array(beam.data[0, 0, :, :], beam2dwcs, model.polarisation_frame)
        reprojected_beam2d, footprint = reproject_image(beam2d, model2dwcs, shape=model2dshape)
        assert numpy.max(footprint.data) > 0.0, "No overlap between beam and model"
        
        reprojected_beam2d.data *= reprojected_beam2d.data
        reprojected_beam2d.data[footprint.data <= 0.0] = 0.0
        for pol in range(npol):
            reprojected_beam.data[chan, pol, :, :] = reprojected_beam2d.data[:, :]
    
    return reprojected_beam
