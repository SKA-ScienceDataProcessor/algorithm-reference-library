"""
Functions to create primary beam and voltage pattern models
"""

import collections
import logging

import numpy
from astropy import constants as const

from data_models.memory_data_models import Image
from data_models.parameters import arl_path
from processing_components.image.operations import create_empty_image_like
from processing_library.image.operations import fft_image, pad_image
from ..image.operations import import_image_from_fits, create_image_from_array, \
    reproject_image

log = logging.getLogger(__name__)


def set_pb_header(pb, use_local=True):
    """Fill in PB header correctly
    
    :param pb:
    :return:
    """
    if use_local:
        nchan, npol, ny, nx = pb.shape
        pb.wcs.wcs.ctype[0] = 'AZELGEO long'
        pb.wcs.wcs.ctype[1] = 'AZELGEO lati'
        pb.wcs.wcs.crval[0] = 0.0
        pb.wcs.wcs.crval[1] = 0.0
        pb.wcs.wcs.crpix[0] = nx // 2
        pb.wcs.wcs.crpix[1] = ny // 2
    
    return pb


def gauss(x0, y0, amp, sigma, rho, diff, a):
    """
    2D gaussian
 
    :param a: Grid of aperture plane coordinates
    """
    dx = a[..., 0] - x0
    dy = a[..., 1] - y0
    r = numpy.hypot(dx, dy)
    return amp * numpy.exp(-1.0 / (2 * sigma ** 2) * (r ** 2 +
                                                      rho * (dx * dy) +
                                                      diff * (dx ** 2 - dy ** 2)))


def ft_disk(r):
    from scipy.special import jn  # pylint: disable=no-name-in-module
    result = numpy.zeros_like(r, dtype='complex')
    result[r > 0] = 2.0 * jn(1, r[r > 0]) / r[r > 0]
    rsmall = 1e-9
    result[r == 0] = 2.0 * jn(1, rsmall) / rsmall
    return result


def tapered_disk(r, radius, blockage=0.0, taper='gaussian', edge=1.0):
    result = numpy.zeros_like(r, dtype='complex')
    if taper == 'gaussian':
        # exp(-gscale*radius**2) = taper
        gscale = -numpy.log(edge) / radius ** 2
        result[r < radius] = numpy.exp(- gscale * r[r < radius] ** 2)
    result[r < blockage] = 0.0
    return result


def create_vp(model, telescope='MID', pointingcentre=None, padding=4, use_local=True):
    """
    Make an image like model and fill it with an analytical model of the voltage pattern
    :param model: Template image
    :param telescope: 'VLA' or 'ASKAP'
    :return: Primary beam image
    """
    if telescope == 'MID_GAUSS':
        log.debug("create_vp: Using numeric tapered Gaussian model for MID voltage pattern")
        
        edge = numpy.power(10, -1.2)
        return create_vp_generic_numeric(model, pointingcentre=pointingcentre, diameter=15.0, blockage=0.0,
                                         edge=edge, padding=padding, use_local=use_local)
    elif telescope == 'MID':
        log.debug("create_vp: Using no taper analytic model for MID voltage pattern")
        return create_vp_generic(model, pointingcentre=pointingcentre, diameter=15.0, blockage=0.0, use_local=use_local)
    elif telescope == 'MID_GRASP':
        log.debug("create_vp: Using GRASP model for MID voltage pattern")
        real_vp = import_image_from_fits(arl_path('data/models/MID_GRASP_VP_real.fits'))
        imag_vp = import_image_from_fits(arl_path('data/models/MID_GRASP_VP_imag.fits'))
        real_vp.data = real_vp.data + 1j * imag_vp.data
        real_vp.data /= numpy.max(numpy.abs(real_vp.data))
        return real_vp
    elif telescope == 'MID_FEKO':
        log.debug("create_vp: Using FEKO model for MID voltage pattern")
        real_vp = import_image_from_fits(arl_path('data/models/MID_FEKO_VP_B2_45_1360_real.fits'))
        imag_vp = import_image_from_fits(arl_path('data/models/MID_FEKO_VP_B2_45_1360_imag.fits'))
        real_vp.data = real_vp.data + 1j * imag_vp.data
        real_vp.data /= numpy.max(numpy.abs(real_vp.data))
        return real_vp
    elif telescope == 'MEERKAT':
        return create_vp_generic(model, pointingcentre=pointingcentre, diameter=13.5, blockage=0.0, use_local=use_local)
    elif telescope[0:3] == 'LOW':
        return create_low_test_vp(model)
    elif telescope[0:3] == 'VLA':
        return create_vp_generic(model, pointingcentre=pointingcentre, diameter=25.0, blockage=1.8, use_local=use_local)
    elif telescope[0:5] == 'ASKAP':
        return create_vp_generic(model, pointingcentre=pointingcentre, diameter=12.0, blockage=1.0, use_local=use_local)
    else:
        raise NotImplementedError('Telescope %s has no voltage pattern model' % telescope)


def create_pb(model, telescope='MID', pointingcentre=None, use_local=True):
    """
    Make an image like model and fill it with an analytical model of the primary beam
    :param model: Template image
    :param telescope: 'VLA' or 'ASKAP'
    :return: Primary beam image
    """
    if telescope == 'LOW':
        beam = create_low_test_beam(model, use_local=use_local)
    else:
        beam = create_vp(model, telescope, pointingcentre, use_local=use_local)
        beam.data = numpy.real(beam.data * numpy.conjugate(beam.data))
    
    set_pb_header(beam, use_local=use_local)
    return beam


def mosaic_pb(model, telescope, pointingcentres, use_local=True):
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
        pb = create_pb(model, telescope, pointingcentre=pc, use_local=use_local)
        sumpb.data += pb.data ** 2
    sumpb.data = numpy.sqrt(sumpb.data)
    return sumpb


def create_pb_generic(model, pointingcentre=None, diameter=25.0, blockage=1.8, use_local=True):
    """
    Make an image like model and fill it with an analytical model of the primary beam
    :param model:
    :return:
    """
    beam = create_vp_generic(model, pointingcentre, diameter, blockage, use_local=use_local)
    beam.data = numpy.real(beam.data * numpy.conjugate(beam.data))
    set_pb_header(beam, use_local=use_local)
    return beam


def create_vp_generic(model, pointingcentre=None, diameter=25.0, blockage=1.8, use_local=True):
    """
    Make an image like model and fill it with an analytical model of the primary beam
    :param model:
    :return:
    """
    beam = create_empty_image_like(model)
    beam.data = numpy.zeros(beam.data.shape, dtype='complex')
    
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
    
    set_pb_header(beam, use_local=use_local)
    return beam


def create_vp_generic_numeric(model, pointingcentre=None, diameter=15.0, blockage=0.0, taper='gaussian',
                              edge=0.03162278, zernikes=None, padding=4, use_local=True, rho=0.0, diff=0.0):
    """
    Make an image like model and fill it with an analytical model of the primary beam
    
    The elements of the analytical model are:
    - dish, optionally blocked
    - Gaussian taper, default is -12dB at the edge
    - Offset to pointing centre (optional)
    - zernikes in a list of dictionaries. Each list element is of the form {"coeff":0.1, "noll":5}. See aotools for
    more details
    - Output image can be in RA, DEC coordinates or AZELGEO coordinates (the default). use_local=True means to use
    AZELGEO coordinates centered on 0deg 0deg.
    
    The dish is zero padded according to padding and FFT'ed to get the voltage pattern.
    
    :param model:
    :param pointingcentre: SkyCoord of desired pointing centre
    :param diameter: Diameter of dish in metres
    :param blockage: Blockage of dish in metres
    :param taper: "Gaussian" or None
    :param edge: Value of taper at the end of the dish (default corresponds to -12dB)
    :param zernikes: Zernikes to be applied as phase across the dish (see above)
    :param padding: Pad the image by this amount
    :param use_local: Use local frame (AZELGEO)?
    :return:
    """
    beam = create_empty_image_like(model)
    nchan, npol, ny, nx = beam.shape
    padded_shape = [nchan, npol, padding * ny, padding * nx]
    padded_beam = pad_image(beam, padded_shape)
    padded_beam.data = numpy.zeros(padded_beam.data.shape, dtype='complex')
    _, _, pny, pnx = padded_beam.shape
    
    xfr = fft_image(padded_beam)
    cx, cy = xfr.wcs.sub(2).wcs.crpix[0] - 1, xfr.wcs.sub(2).wcs.crpix[1] - 1
    
    for chan in range(nchan):
        
        # The frequency axis is the second to last in the beam
        frequency = xfr.wcs.sub(['spectral']).wcs_pix2world([chan], 0)[0]
        wavelength = const.c.to('m s^-1').value / frequency
        
        scalex = xfr.wcs.sub(2).wcs.cdelt[0] * wavelength
        scaley = xfr.wcs.sub(2).wcs.cdelt[1] * wavelength
        # xx, yy in metres
        xx, yy = numpy.meshgrid(scalex * (range(pnx) - cx), scaley * (range(pny) - cy))
        
        # rr in metres
        rr = numpy.sqrt(xx**2 + yy**2)
        for pol in range(npol):
            xfr.data[chan, pol, ...] = tapered_disk(rr, diameter / 2.0, blockage=blockage / 2.0, edge=edge,
                                                    taper=taper)
        
        if pointingcentre is not None:
            # Correct for pointing centre
            pcx, pcy = pointingcentre.to_pixel(padded_beam.wcs, origin=0)
            pxx, pyy = numpy.meshgrid((range(pnx) - cx), (range(pny) - cy))
            phase = 2 * numpy.pi * ((pcx - cx) * pxx / float(pnx) + (pcy - cy) * pyy / float(pny))
            for pol in range(npol):
                xfr.data[chan, pol, ...] *= numpy.exp(1j * phase)
        
        if isinstance(zernikes, collections.Iterable):
            try:
                import aotools
            except ModuleNotFoundError:
                raise ModuleNotFoundError("aotools is not installed")
            
            ndisk = numpy.ceil(numpy.abs(diameter / scalex)).astype('int')[0]
            ndisk = 2 * ((ndisk + 1) // 2)
            phase = numpy.zeros([ndisk, ndisk])
            for zernike in zernikes:
                phase = zernike['coeff'] * aotools.functions.zernike(zernike['noll'], ndisk)
            
            # import matplotlib.pyplot as plt
            # plt.clf()
            # plt.imshow(phase)
            # plt.colorbar()
            # plt.show()
            #
            blc = pnx // 2 - ndisk // 2
            trc = pnx // 2 + ndisk // 2
            for pol in range(npol):
                xfr.data[chan, pol, blc:trc, blc:trc] = xfr.data[chan, pol, blc:trc, blc:trc] * numpy.exp(1j * phase)
    
    padded_beam = fft_image(xfr, padded_beam)
    
    # Undo padding
    beam = create_empty_image_like(model)
    beam.data = padded_beam.data[..., (pny // 2 - ny // 2):(pny // 2 + ny // 2),
                (pnx // 2 - nx // 2):(pnx // 2 + nx // 2)]
    for chan in range(nchan):
        beam.data[chan, ...] /= numpy.max(numpy.abs(beam.data[chan, ...]))
    
    set_pb_header(beam, use_local=use_local)
    return beam


def create_low_test_beam(model: Image, use_local=True) -> Image:
    """Create a test power beam for LOW using an image from OSKAR

    :param model: Template image
    :return: Image
    """
    beam = import_image_from_fits(arl_path('data/models/SKA1_LOW_beam.fits'))
    
    # Scale the image cellsize to account for the different in frequencies. Eventually we will want to
    # use a frequency cube
    log.debug("create_low_test_beam: LOW voltage pattern is defined at %.3f MHz" % (beam.wcs.wcs.crval[2] * 1e-6))
    
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
        
        reprojected_beam2d.data[footprint.data <= 0.0] = 0.0
        for pol in range(npol):
            reprojected_beam.data[chan, pol, :, :] = reprojected_beam2d.data[:, :]
    
    set_pb_header(reprojected_beam, use_local=use_local)
    return reprojected_beam


def create_low_test_vp(model: Image, use_local=True) -> Image:
    """Create a test power beam for LOW using an image from OSKAR

    :param model: Template image
    :return: Image
    """
    
    # TODO: Get true voltage beam from OSKAR
    beam = import_image_from_fits(arl_path('data/models/SKA1_LOW_beam.fits'))
    beam.data = numpy.sqrt(beam.data).astype('complex')
    
    # Scale the image cellsize to account for the different in frequencies. Eventually we will want to
    # use a frequency cube
    log.debug("create_low_test_beam: LOW voltage pattern is defined at %.3f MHz" % (beam.wcs.wcs.crval[2] * 1e-6))
    
    nchan, npol, ny, nx = model.shape
    
    # We need to interpolate each frequency channel separately. The beam is assumed to just scale with
    # frequency.
    
    reprojected_beam = create_empty_image_like(model)
    reprojected_beam.data = reprojected_beam.data.astype('complex')
    
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
        
        beam2d_real = create_image_from_array(numpy.real(beam.data[0, 0, :, :]), beam2dwcs, model.polarisation_frame)
        beam2d_imag = create_image_from_array(numpy.imag(beam.data[0, 0, :, :]), beam2dwcs, model.polarisation_frame)
        reprojected_beam2d_real, footprint = reproject_image(beam2d_real, model2dwcs, shape=model2dshape)
        reprojected_beam2d_imag, footprint = reproject_image(beam2d_imag, model2dwcs, shape=model2dshape)
        assert numpy.max(footprint.data) > 0.0, "No overlap between beam and model"
        
        reprojected_beam2d_real.data[footprint.data <= 0.0] = 0.0
        reprojected_beam2d_imag.data[footprint.data <= 0.0] = 0.0
        for pol in range(npol):
            reprojected_beam.data[chan, pol, :, :] = reprojected_beam2d_real.data[:, :] \
                                                     + 1j * reprojected_beam2d_imag.data[:, :]
    
    set_pb_header(reprojected_beam, use_local=use_local)
    return reprojected_beam
