import copy
import warnings

import numpy
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS
from astropy.wcs.utils import skycoord_to_pixel
from reproject import reproject_interp

from data_models.memory_data_models import Image, QA
from data_models.parameters import arl_path

import logging
log = logging.getLogger(__name__)

from libs.image.operations import image_sizeof, polarisation_frame_from_wcs, create_image_from_array, checkwcs, \
    copy_image, create_empty_image_like

from data_models.polarisation import PolarisationFrame, convert_stokes_to_linear, convert_stokes_to_circular, \
    convert_linear_to_stokes, convert_circular_to_stokes


def export_image_to_fits(im: Image, fitsfile: str = 'imaging.fits'):
    """ Write an image to fits
    
    :param im: Image
    :param fitsfile: Name of output fits file
    """
    assert isinstance(im, Image), im
    return fits.writeto(filename=fitsfile, data=im.data, header=im.wcs.to_header(), overwrite=True)


def import_image_from_fits(fitsfile: str) -> Image:
    """ Read an Image from fits
    
    :param fitsfile:
    :return: Image
    """
    fim = Image()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FITSFixedWarning)
        hdulist = fits.open(arl_path(fitsfile))
        fim.data = hdulist[0].data
        fim.wcs = WCS(arl_path(fitsfile))
        hdulist.close()
    
    if len(fim.data) == 2:
        fim.polarisation_frame = PolarisationFrame('stokesI')
    else:
        try:
            fim.polarisation_frame = polarisation_frame_from_wcs(fim.wcs, fim.data.shape)
        except ValueError:
            fim.polarisation_frame = PolarisationFrame('stokesI')
    
    log.debug("import_image_from_fits: created %s image of shape %s, size %.3f (GB)" %
              (fim.data.dtype, str(fim.shape), image_sizeof(fim)))
    log.debug("import_image_from_fits: Max, min in %s = %.6f, %.6f" % (fitsfile, fim.data.max(), fim.data.min()))
    
    assert isinstance(fim, Image)
    return fim


def reproject_image(im: Image, newwcs: WCS, shape=None) -> (Image, Image):
    """ Re-project an image to a new coordinate system
    
    Currently uses the reproject python package. This seems to have some features do be careful using this method.
    For timeslice imaging I had to use griddata.


    :param im: Image to be reprojected
    :param newwcs: New WCS
    :param shape:
    :return: Reprojected Image, Footprint Image
    """
    
    assert isinstance(im, Image), im
    rep, foot = reproject_interp((im.data, im.wcs), newwcs, shape, order='bicubic',
                                 independent_celestial_slices=True)
    return create_image_from_array(rep, newwcs, im.polarisation_frame), create_image_from_array(foot, newwcs,
                                                                                                im.polarisation_frame)


def add_image(im1: Image, im2: Image, docheckwcs=False) -> Image:
    """ Add two images
    
    :param docheckwcs:
    :param im1:
    :param im2:
    :return: Image
    """
    assert isinstance(im1, Image), im1
    assert isinstance(im2, Image), im2
    if docheckwcs:
        checkwcs(im1.wcs, im2.wcs)
    
    assert im1.polarisation_frame == im2.polarisation_frame
    
    return create_image_from_array(im1.data + im2.data, im1.wcs, im1.polarisation_frame)


def qa_image(im, context="") -> QA:
    """Assess the quality of an image

    :param im:
    :return: QA
    """
    assert isinstance(im, Image), im
    data = {'shape': str(im.data.shape),
            'max': numpy.max(im.data),
            'min': numpy.min(im.data),
            'rms': numpy.std(im.data),
            'sum': numpy.sum(im.data),
            'medianabs': numpy.median(numpy.abs(im.data)),
            'median': numpy.median(im.data)}
    
    qa = QA(origin="qa_image", data=data, context=context)
    return qa


def show_image(im: Image, fig=None, title: str = '', pol=0, chan=0, cm='rainbow', components=None,
               vmin=None, vmax=None):
    """ Show an Image with coordinates using matplotlib, optionally with components

    :param im:
    :param fig:
    :param title:
    :param pol: Polarisation
    :param chan: Channel
    :param components: Optional components
    :param vmin: Clip to this minimum
    :param vmax: Clip to this maximum
    :return:
    """
    import matplotlib.pyplot as plt
    
    assert isinstance(im, Image), im
    if not fig:
        fig = plt.figure()
    plt.clf()
    fig.add_subplot(111, projection=im.wcs.sub(['longitude', 'latitude']))
    if len(im.data.shape) == 4:
        data_array = numpy.real(im.data[chan, pol, :, :])
    else:
        data_array = numpy.real(im.data)
        
    if vmax is None:
        vmax = numpy.max(data_array)
    if vmin is None:
        vmin = numpy.min(data_array)

    plt.imshow(data_array, origin = 'lower', cmap = cm, vmax=vmax, vmin=vmin)

    plt.xlabel('RA---SIN')
    plt.ylabel('DEC--SIN')
    plt.title(title)
    plt.colorbar()
    
    if components is not None:
        for sc in components:
            x, y = skycoord_to_pixel(sc.direction, im.wcs, 1, 'wcs')
            plt.plot(x, y, marker='+', color='red')
    return fig


def smooth_image(model: Image, width=1.0):
    """ Smooth an image with a kernel
    
    """
    import astropy.convolution
    
    assert isinstance(model, Image), model
    kernel = astropy.convolution.kernels.Gaussian2DKernel(width)
    
    cmodel = create_empty_image_like(model)
    nchan, npol, _, _ = model.shape
    for pol in range(npol):
        for chan in range(nchan):
            cmodel.data[chan, pol, :, :] = astropy.convolution.convolve(model.data[chan, pol, :, :], kernel,
                                                                        normalize_kernel=False)
    if isinstance(kernel, astropy.convolution.kernels.Gaussian2DKernel):
        cmodel.data *= 2 * numpy.pi * width ** 2
    
    return cmodel


def calculate_image_frequency_moments(im: Image, reference_frequency=None, nmoments=3) -> Image:
    """Calculate frequency weighted moments
    
    Weights are ((freq-reference_frequency)/reference_frequency)**moment
    
    Note that the spectral axis is replaced by a MOMENT axis.
    
    For example, to find the moments and then reconstruct from just the moments::
    
        moment_cube = calculate_image_frequency_moments(model_multichannel, nmoments=5)
        reconstructed_cube = calculate_image_from_frequency_moments(model_multichannel, moment_cube)

    :param im: Image cube
    :param reference_frequency: Reference frequency (default None uses average)
    :param nmoments: Number of moments to calculate
    :return: Moments image
    """
    assert isinstance(im, Image)
    nchan, npol, ny, nx = im.shape
    channels = numpy.arange(nchan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FITSFixedWarning)
        freq = im.wcs.sub(['spectral']).wcs_pix2world(channels, 0)[0]
    
    assert nmoments <= nchan, "Number of moments %d cannot exceed the number of channels %d" % (nmoments, nchan)
    
    if reference_frequency is None:
        reference_frequency = numpy.average(freq)
    log.info("calculate_image_frequency_moments: Reference frequency = %.3f (MHz)" % (reference_frequency))
    
    moment_data = numpy.zeros([nmoments, npol, ny, nx])
    
    for moment in range(nmoments):
        for chan in range(nchan):
            weight = numpy.power((freq[chan] - reference_frequency) / reference_frequency, moment)
            moment_data[moment, ...] += im.data[chan, ...] * weight
    
    moment_wcs = copy.deepcopy(im.wcs)
    moment_wcs.wcs.ctype[3] = 'MOMENT'
    moment_wcs.wcs.crval[3] = 0.0
    moment_wcs.wcs.crpix[3] = 1.0
    moment_wcs.wcs.cdelt[3] = 1.0
    moment_wcs.wcs.cunit[3] = ''
    
    return create_image_from_array(moment_data, moment_wcs, im.polarisation_frame)


def calculate_image_from_frequency_moments(im: Image, moment_image: Image, reference_frequency=None) -> Image:
    """Calculate image from frequency weighted moments

    Weights are ((freq-reference_frequency)/reference_frequency)**moment

    Note that a new image is created
    
    For example, to find the moments and then reconstruct from just the moments::
    
        moment_cube = calculate_image_frequency_moments(model_multichannel, nmoments=5)
        reconstructed_cube = calculate_image_from_frequency_moments(model_multichannel, moment_cube)


    :param im: Image cube to be reconstructed
    :param moment_image: Moment cube (constructed using calculate_image_frequency_moments)
    :param reference_frequency: Reference frequency (default None uses average)
    :return: reconstructed image
    """
    assert isinstance(im, Image)
    nchan, npol, ny, nx = im.shape
    nmoments, mnpol, mny, mnx = moment_image.shape
    
    assert npol == mnpol
    assert ny == mny
    assert nx == mnx
    
    assert moment_image.wcs.wcs.ctype[3] == 'MOMENT', "Second image should be a moment image"
    
    channels = numpy.arange(nchan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FITSFixedWarning)
        freq = im.wcs.sub(['spectral']).wcs_pix2world(channels, 0)[0]
    
    if reference_frequency is None:
        reference_frequency = numpy.average(freq)
    log.info("calculate_image_from_frequency_moments: Reference frequency = %.3f (MHz)" % (reference_frequency))
    
    newim = copy_image(im)
    
    newim.data[...] = 0.0
    
    for moment in range(nmoments):
        for chan in range(nchan):
            weight = numpy.power((freq[chan] - reference_frequency) / reference_frequency, moment)
            newim.data[chan, ...] += moment_image.data[moment, ...] * weight
    
    return newim


def remove_continuum_image(im: Image, degree=1, mask=None):
    """ Fit and remove continuum visibility in place
    
    Fit a polynomial in frequency of the specified degree where mask is True

    :param im:
    :param degree: 1 is a constant, 2 is a slope, etc.
    :param mask:
    :return:
    """
    assert isinstance(im, Image)
    
    if mask is not None:
        assert numpy.sum(mask) > 2 * degree, "Insufficient channels for fit"
    
    nchan, npol, ny, nx = im.shape
    channels = numpy.arange(nchan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FITSFixedWarning)
        frequency = im.wcs.sub(['spectral']).wcs_pix2world(channels, 0)[0]
    frequency -= frequency[nchan // 2]
    frequency /= numpy.max(frequency)
    wt = numpy.ones_like(frequency)
    if mask is not None:
        wt[mask] = 0.0
    
    for pol in range(npol):
        for y in range(ny):
            for x in range(nx):
                fit = numpy.polyfit(frequency, im.data[:, pol, y, x], w=wt, deg=degree)
                prediction = numpy.polyval(fit, frequency)
                im.data[:, pol, y, x] -= prediction
    return im


def convert_stokes_to_polimage(im: Image, polarisation_frame: PolarisationFrame):
    """Convert a stokes image to polarisation_frame

    """
    
    assert isinstance(im, Image)
    assert isinstance(polarisation_frame, PolarisationFrame)
    
    if polarisation_frame == PolarisationFrame('linear'):
        cimarr = convert_stokes_to_linear(im.data)
        return create_image_from_array(cimarr, im.wcs, polarisation_frame)
    elif polarisation_frame == PolarisationFrame('circular'):
        cimarr = convert_stokes_to_circular(im.data)
        return create_image_from_array(cimarr, im.wcs, polarisation_frame)
    else:
        raise ValueError("Cannot convert stokes to %s" % (polarisation_frame.type))


def convert_polimage_to_stokes(im: Image):
    """Convert a polarisation image to stokes (complex)
    
    """
    assert isinstance(im, Image)
    assert im.data.dtype == 'complex'
    
    if im.polarisation_frame == PolarisationFrame('linear'):
        cimarr = convert_linear_to_stokes(im.data)
        return create_image_from_array(cimarr, im.wcs, PolarisationFrame('stokesIQUV'))
    elif im.polarisation_frame == PolarisationFrame('circular'):
        cimarr = convert_circular_to_stokes(im.data)
        return create_image_from_array(cimarr, im.wcs, PolarisationFrame('stokesIQUV'))
    else:
        raise ValueError("Cannot convert %s to stokes" % (im.polarisation_frame.type))


