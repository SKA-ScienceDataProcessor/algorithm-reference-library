
#
"""
Functions that define and manipulate images. Images are just data and a World Coordinate System.
"""

import numpy
import os
import copy
import warnings

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from reproject import reproject_interp

from arl.data.parameters import arl_path
from arl.data.data_models import Image, QA
from arl.data.polarisation import PolarisationFrame, convert_circular_to_stokes, convert_linear_to_stokes, \
    convert_stokes_to_circular, convert_stokes_to_linear
from arl.fourier_transforms.fft_support import fft, ifft, pad_mid, extract_mid


import logging

from arl.fourier_transforms.convolutional_gridding import w_beam
from arl.data.parameters import get_parameter

log = logging.getLogger(__name__)


def image_sizeof(im: Image):
    """ Return size in GB
    """
    return im.size()


def create_image_from_array(data: numpy.array, wcs: WCS = None,
                            polarisation_frame=PolarisationFrame('stokesI')) -> Image:
    """ Create an image from an array and optional wcs

    :param data: Numpy.array
    :param wcs: World coordinate system
    :param polarisation_frame: Polarisation Frame
    :return: Image
    
    """
    fim = Image()
    fim.polarisation_frame = polarisation_frame
    
    fim.data = data
    if wcs is None:
        fim.wcs = None
    else:
        fim.wcs = wcs.deepcopy()
    
    if image_sizeof(fim) >= 1.0:
        log.debug("create_image_from_array: created %s image of shape %s, size %.3f (GB)" %
                  (fim.data.dtype, str(fim.shape), image_sizeof(fim)))
        
    assert type(fim) == Image, "Type is %s" % type(fim)
    return fim


def copy_image(im: Image) -> Image:
    """ Create an image from an array
    
    Performs deepcopy of data, breaking reference semantics

    :param im:
    :return: Image
    
    """
    assert type(im) == Image, "Type is %s" % type(im)
    fim = Image()
    fim.polarisation_frame = im.polarisation_frame
    fim.data = copy.deepcopy(im.data)
    if im.wcs is None:
        fim.wcs = None
    else:
        fim.wcs = copy.deepcopy(im.wcs)
    if image_sizeof(fim) >= 1.0:
        log.debug("copy_image: copied %s image of shape %s, size %.3f (GB)" %
                  (fim.data.dtype, str(fim.shape), image_sizeof(fim)))
    assert type(fim) == Image
    return fim


def create_empty_image_like(im: Image) -> Image:
    """ Create an empty image like another in shape and wcs

    :param im:
    :return: Image
    
    """
    assert type(im) == Image, "Type is %s" % type(im)
    fim = Image()
    fim.polarisation_frame = im.polarisation_frame
    fim.data = numpy.zeros_like(im.data)
    if im.wcs is None:
        fim.wcs = None
    else:
        fim.wcs = copy.deepcopy(im.wcs)
    if image_sizeof(im) >= 1.0:
        log.debug("create_empty_image_like: created %s image of shape %s, size %.3f (GB)" %
                  (fim.data.dtype, str(fim.shape), image_sizeof(fim)))
    assert type(fim) == Image, "Type is %s" % type(fim)
    return fim


def polarisation_frame_from_wcs(wcs, shape) -> PolarisationFrame:
    """Convert wcs to polarisation_frame

    See FITS definition in Table 29 of https://fits.gsfc.nasa.gov/standard40/fits_standard40draft1.pdf
    or subsequent revision

        1 I Standard Stokes unpolarized
        2 Q Standard Stokes linear
        3 U Standard Stokes linear
        4 V Standard Stokes circular
        −1 RR Right-right circular
        −2 LL Left-left circular
        −3 RL Right-left cross-circular
        −4 LR Left-right cross-circular
        −5 XX X parallel linear
        −6 YY Y parallel linear
        −7 XY XY cross linear
        −8 YX YX cross linear

        stokesI [1]
        stokesIQUV [1,2,3,4]
        circular [-1,-2,-3,-4]
        linear [-5,-6,-7,-8]

    """
    # The third axis should be stokes:
    
    polarisation_frame = None
    
    if len(shape) == 2:
        polarisation_frame = PolarisationFrame("stokesI")
    else:
        npol = shape[1]
        pol = wcs.sub(['stokes']).wcs_pix2world(range(npol), 0)[0]
        pol = numpy.array(pol, dtype='int')
        for key in PolarisationFrame.fits_codes.keys():
            keypol = numpy.array(PolarisationFrame.fits_codes[key])
            if numpy.array_equal(pol, keypol):
                polarisation_frame = PolarisationFrame(key)
                return polarisation_frame
    if polarisation_frame is None:
        raise ValueError("Cannot determine polarisation code")
    
    assert type(polarisation_frame) == PolarisationFrame
    return polarisation_frame


def export_image_to_fits(im: Image, fitsfile: str = 'imaging.fits'):
    """ Write an image to fits
    
    :param im: Image
    :param fitsfile: Name of output fits file
    """
    assert type(im) == Image
    return fits.writeto(filename=fitsfile, data=im.data, header=im.wcs.to_header(), overwrite=True)


def import_image_from_fits(fitsfile: str, mute_warnings=True) -> Image:
    """ Read an Image from fits
    
    :param fitsfile:
    :return: Image
    """
    fim = Image()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hdulist = fits.open(arl_path(fitsfile))
        fim.data = hdulist[0].data
        fim.wcs = WCS(arl_path(fitsfile))
        hdulist.close()
    
    if len(fim.data) == 2:
        fim.polarisation_frame = PolarisationFrame('stokesI')
    else:
        try:
            fim.polarisation_frame = polarisation_frame_from_wcs(fim.wcs, fim.data.shape)
        except:
            fim.polarisation_frame = PolarisationFrame('stokesI')
    
    log.debug("import_image_from_fits: created %s image of shape %s, size %.3f (GB)" %
              (fim.data.dtype, str(fim.shape), image_sizeof(fim)))
    log.debug("import_image_from_fits: Max, min in %s = %.6f, %.6f" % (fitsfile, fim.data.max(), fim.data.min()))

    assert type(fim) == Image
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
    
    assert type(im) == Image
    rep, foot = reproject_interp((im.data, im.wcs), newwcs, shape, order='bicubic',
                                 independent_celestial_slices=True)
    return create_image_from_array(rep, newwcs), create_image_from_array(foot, newwcs)


def checkwcs(wcs1, wcs2):
    """ Check for compatbility of wcs
    
    :param wcs1:
    :param wcs2:
    """
    pass
    # No confidence in this next test
    # assert wcs1.wcs.compare(wcs2.wcs, cmp=1 | 2 | 4), "WCS's do not agree"


def add_image(im1: Image, im2: Image, docheckwcs=False) -> Image:
    """ Add two images
    
    :param docheckwcs:
    :param im1:
    :param im2:
    :return: Image
    """
    assert type(im1) == Image
    assert type(im2) == Image
    if docheckwcs:
        checkwcs(im1.wcs, im2.wcs)
    
    assert im1.polarisation_frame == im2.polarisation_frame
    
    return create_image_from_array(im1.data + im2.data, im1.wcs, im1.polarisation_frame)


def qa_image(im, mask=None, **kwargs) -> QA:
    """Assess the quality of an image

    :param params:
    :param im:
    :return: QA
    """
    assert type(im) == Image
    if mask is None:
        data = {'shape': str(im.data.shape),
                'max': numpy.max(im.data),
                'min': numpy.min(im.data),
                'rms': numpy.std(im.data),
                'sum': numpy.sum(im.data),
                'medianabs': numpy.median(numpy.abs(im.data)),
                'median': numpy.median(im.data)}
    else:
        mdata = im.data[mask.data > 0.0]
        data = {'shape': str(im.data.shape),
                'max': numpy.max(mdata),
                'min': numpy.min(mdata),
                'rms': numpy.std(mdata),
                'sum': numpy.sum(mdata),
                'medianabs': numpy.median(numpy.abs(mdata)),
                'median': numpy.median(mdata)}
    
    qa = QA(origin="qa_image",
            data=data,
            context=get_parameter(kwargs, 'context', ""))
    return qa


def show_image(im: Image, fig=None, title: str = '', pol=0, chan=0, cm='rainbow'):
    """ Show an Image with coordinates using matplotlib

    :param im:
    :param fig:
    :param title:
    :return:
    """
    import matplotlib.pyplot as plt

    assert type(im) == Image
    if not fig:
        fig = plt.figure()
    plt.clf()
    fig.add_subplot(111, projection=im.wcs.sub(['longitude', 'latitude']))
    if len(im.data.shape) == 4:
        plt.imshow(numpy.real(im.data[chan, pol, :, :]), origin='lower', cmap=cm)
    elif len(im.data.shape) == 2:
        plt.imshow(numpy.real(im.data[:, :]), origin='lower', cmap=cm)
    plt.xlabel('RA---SIN')
    plt.ylabel('DEC--SIN')
    plt.title(title)
    plt.colorbar()
    return fig


def convert_stokes_to_polimage(im: Image, polarisation_frame: PolarisationFrame):
    """Convert a stokes image to polarisation_frame

    """
    
    assert type(im) == Image
    assert type(polarisation_frame) == PolarisationFrame
    
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
    assert type(im) == Image
    assert im.data.dtype == 'complex'
    
    if im.polarisation_frame == PolarisationFrame('linear'):
        cimarr = convert_linear_to_stokes(im.data)
        return create_image_from_array(cimarr, im.wcs, PolarisationFrame('stokesIQUV'))
    elif im.polarisation_frame == PolarisationFrame('circular'):
        cimarr = convert_circular_to_stokes(im.data)
        return create_image_from_array(cimarr, im.wcs, PolarisationFrame('stokesIQUV'))
    else:
        raise ValueError("Cannot convert %s to stokes" % (im.polarisation_frame.type))


def smooth_image(model: Image, width=1.0):
    """ Smooth an image with a kernel
    
    """
    import astropy.convolution
    
    assert type(model) == Image
    kernel = astropy.convolution.kernels.Gaussian2DKernel(width)
    
    cmodel = create_empty_image_like(model)
    nchan, npol, _, _ = model.shape
    for pol in range(npol):
        for chan in range(nchan):
            cmodel.data[chan, pol, :, :] = astropy.convolution.convolve(model.data[chan, pol, :, :], kernel,
                                                                  normalize_kernel=False)
    if type(kernel) is astropy.convolution.kernels.Gaussian2DKernel:
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
    assert type(im) == Image
    nchan, npol, ny, nx = im.shape
    channels = numpy.arange(nchan)
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
    assert type(im) == Image
    nchan, npol, ny, nx = im.shape
    nmoments, mnpol, mny, mnx = moment_image.shape
    
    assert npol == mnpol
    assert ny == mny
    assert nx == mnx
    
    assert moment_image.wcs.wcs.ctype[3] == 'MOMENT', "Second image should be a moment image"
    
    channels = numpy.arange(nchan)
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


def remove_continuum_image(im: Image, degree=1, mask=None, **kwargs):
    """ Fit and remove continuum visibility in place
    
    Fit a polynomial in frequency of the specified degree where mask is True

    :param im:
    :param deg:
    :param mask:
    :param kwargs:
    :return:
    """
    assert type(im) == Image

    if mask is not None:
        assert numpy.sum(mask) > 2 * degree, "Insufficient channels for fit"

    nchan, npol, ny, nx = im.shape
    channels = numpy.arange(nchan)
    frequency = im.wcs.sub(['spectral']).wcs_pix2world(channels, 0)[0]
    frequency -= frequency[nchan//2]
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


def fft_image(im, template_image=None, **kwargs):
    """ FFT an image, transform WCS as well
    
    Prefer to use axes 'UU---SIN' and 'VV---SIN' but astropy will not accept.
    
    :param im:
    :param template:
    :param kwargs:
    :return:
    """
    assert len(im.shape) == 4
    d2r = numpy.pi / 180.0
    ft_wcs = copy.deepcopy(im.wcs)
    ft_shape = im.shape
    if im.wcs.wcs.ctype[0] == 'RA---SIN' and im.wcs.wcs.ctype[1] == 'DEC--SIN':
        ft_wcs.wcs.axis_types[0] = 0
        ft_wcs.wcs.axis_types[1] = 0
        ft_wcs.wcs.crval[0] = 0.0
        ft_wcs.wcs.crval[1] = 0.0
        ft_wcs.wcs.crpix[0] = ft_shape[3] // 2 + 1
        ft_wcs.wcs.crpix[1] = ft_shape[2] // 2 + 1
        ft_wcs.wcs.ctype[0] = 'UU'
        ft_wcs.wcs.ctype[1] = 'VV'
        ft_wcs.wcs.cdelt[0] = 1.0 / (ft_shape[3] * d2r * im.wcs.wcs.cdelt[0])
        ft_wcs.wcs.cdelt[1] = 1.0 / (ft_shape[2] * d2r * im.wcs.wcs.cdelt[1])
        ft_data = ifft(im.data.astype('complex'))
        return create_image_from_array(ft_data, wcs=ft_wcs)
    elif im.wcs.wcs.ctype[0] == 'UU' and im.wcs.wcs.ctype[1] == 'VV':
        ft_wcs.wcs.crval[0] = template_image.wcs.wcs.crval[0]
        ft_wcs.wcs.crval[1] = template_image.wcs.wcs.crval[1]
        ft_wcs.wcs.crpix[0] = template_image.wcs.wcs.crpix[0]
        ft_wcs.wcs.crpix[0] = template_image.wcs.wcs.crpix[1]
        ft_wcs.wcs.ctype[0] = template_image.wcs.wcs.ctype[0]
        ft_wcs.wcs.ctype[1] = template_image.wcs.wcs.ctype[1]
        ft_wcs.wcs.cdelt[0] = template_image.wcs.wcs.cdelt[0]
        ft_wcs.wcs.cdelt[1] = template_image.wcs.wcs.cdelt[1]
        ft_data = fft(im.data.astype('complex'))
        return create_image_from_array(ft_data, wcs=ft_wcs)
    else:
        raise NotImplementedError("Cannot FFT specified axes")

def pad_image(im: Image, shape, **kwargs):
    """Pad an image to desired shape
    
    The wcs crpix is adjusted appropriately
    
    :param im:
    :param shape:
    :param kwargs:
    :return:
    """
    if im.shape == shape:
        return im
    else:
        newwcs = copy.deepcopy(im.wcs)
        newwcs.wcs.crpix[0] = im.wcs.wcs.crpix[0] + shape[3] // 2 - im.shape[3] // 2
        newwcs.wcs.crpix[1] = im.wcs.wcs.crpix[1] + shape[2] // 2 - im.shape[2] // 2
        
        for axis, _ in enumerate(im.shape):
            if shape[axis] < im.shape[axis]:
                raise ValueError("Padded shape %s is smaller than input shape %s" % (shape, im.shape))
        
        newdata = numpy.zeros(shape, dtype=im.data.dtype)
        ystart = shape[2] // 2 - im.shape[2] // 2
        yend = ystart + im.shape[2]
        xstart = shape[3] // 2 - im.shape[3] // 2
        xend = xstart + im.shape[3]
        newdata[...,ystart:yend,xstart:xend] = im.data[...]
        return create_image_from_array(newdata, newwcs, polarisation_frame=im.polarisation_frame)

def convert_image_to_kernel(im: Image, oversampling, kernelwidth, **kwargs):
    """ Convert an image to a gridding kernel
    
    :param im: Image to be converted
    :param oversampling: Oversampling of Image spatially
    :param kernelwidth: Kernel width to be extracted
    :param kwargs:
    :return: numpy.ndarray[nchan, npol, oversampling, oversampling, kernelwidth, kernelwidth]
    """
    naxis = len(im.shape)
    assert naxis == 4
    
    assert numpy.max(numpy.abs(im.data)) > 0.0, "Image is empty"
    
    nchan, npol, ny, nx = im.shape
    assert nx % oversampling == 0, "Oversampling must be a factor of nx"
    assert ny % oversampling == 0, "Oversampling must be a factor of ny"
    
    assert kernelwidth < nx and kernelwidth < ny, "Specified kernel width %d too large"

    assert im.wcs.wcs.ctype[0] == 'UU', 'Axis type %s inappropriate for construction of kernel' % im.wcs.wcs.ctype[0]
    assert im.wcs.wcs.ctype[1] == 'VV', 'Axis type %s inappropriate for construction of kernel' % im.wcs.wcs.ctype[1]
    newwcs = WCS(naxis=naxis+2)
    for axis in range(2):
        newwcs.wcs.ctype[axis] = im.wcs.wcs.ctype[axis]
        newwcs.wcs.crpix[axis] = kernelwidth // 2
        newwcs.wcs.crval[axis] = 0.0
        newwcs.wcs.cdelt[axis] = im.wcs.wcs.cdelt[axis] * oversampling

        newwcs.wcs.ctype[axis+2] = im.wcs.wcs.ctype[axis]
        newwcs.wcs.crpix[axis+2] = oversampling // 2
        newwcs.wcs.crval[axis+2] = 0.0
        newwcs.wcs.cdelt[axis+2] = im.wcs.wcs.cdelt[axis]
        
        # Now do Stokes and Frequency
        newwcs.wcs.ctype[axis+4] = im.wcs.wcs.ctype[axis+2]
        newwcs.wcs.crpix[axis+4] = im.wcs.wcs.crpix[axis+2]
        newwcs.wcs.crval[axis+4] = im.wcs.wcs.crval[axis+2]
        newwcs.wcs.cdelt[axis+4] = im.wcs.wcs.cdelt[axis+2]
        
    newdata_shape = []
    newdata_shape.append(nchan)
    newdata_shape.append(npol)
    newdata_shape.append(oversampling)
    newdata_shape.append(oversampling)
    newdata_shape.append(kernelwidth)
    newdata_shape.append(kernelwidth)
    
    newdata = numpy.zeros(newdata_shape, dtype=im.data.dtype)
    
    assert oversampling * kernelwidth < ny
    assert oversampling * kernelwidth < nx

    ystart = ny // 2 - oversampling * kernelwidth // 2
    xstart = nx // 2 - oversampling * kernelwidth // 2
    yend =   ny // 2 + oversampling * kernelwidth // 2
    xend =   nx // 2 + oversampling * kernelwidth // 2
    for chan in range(nchan):
        for pol in range(npol):
            for y in range(oversampling):
                slicey = slice(yend+y, ystart+y, -oversampling)
                for x in range(oversampling):
                    slicex = slice(xend+x, xstart+x, -oversampling)
                    newdata[chan,pol,y,x,...]=im.data[chan,pol,slicey,slicex]
                    
    return create_image_from_array(newdata, newwcs)


def create_w_term_like(im: Image, w, phasecentre=None, **kwargs) -> Image:
    """Create an image with a w term phase term in it:
    
    .. math::

    I(l,m) = e^{-2 \\pi j (w(\\sqrt{1-l^2-m^2}-1)}

    
    The vis phasecentre is used as the delay centre for the w term (i.e. where n==0)

    :param phasecentre:
    :param im: template image
    :param w: w value to evaluate (default is median abs)
    :return: Image
    """
    
    fim = copy_image(im)
    cellsize = abs(fim.wcs.wcs.cdelt[0]) * numpy.pi / 180.0
    nchan, npol, _, npixel = fim.data.shape
    if phasecentre is SkyCoord:
        wcentre = phasecentre.to_pixel(im.wcs, origin=1)
    else:
        wcentre=[im.wcs.wcs.crpix[0], im.wcs.wcs.crpix[1]]
        
    fim.data = numpy.zeros(fim.shape, dtype='complex')
    remove_shift=get_parameter(kwargs, "remove_shift", False)
    for chan in range(nchan):
        for pol in range(npol):
            fim.data[chan, pol,...] = w_beam(npixel, npixel * cellsize, w=w, cx=wcentre[0], cy=wcentre[1],
                                             remove_shift=remove_shift)
    
    fov = npixel * cellsize
    fresnel = numpy.abs(w) * (0.5 * fov) ** 2
    log.debug('create_w_term_image: For w = %.1f, field of view = %.6f, Fresnel number = %.2f' % (w, fov, fresnel))
    
    return fim