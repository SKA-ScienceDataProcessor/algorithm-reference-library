#
"""
Functions that define and manipulate images. Images are just data and a World Coordinate System.
"""
import copy
import logging
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', FITSFixedWarning)

import numpy

import astropy.units as u
from astropy.coordinates import SkyCoord

from astropy.wcs import WCS

from data_models.polarisation import PolarisationFrame
from data_models.memory_data_models import Image

from processing_library.fourier_transforms.convolutional_gridding import w_beam
from processing_library.fourier_transforms.fft_support import ifft, fft

log = logging.getLogger(__name__)


def image_sizeof(im: Image):
    """ Return size in GB
    """
    return im.size()


def create_image(npixel=512, cellsize=0.000015, polarisation_frame=PolarisationFrame("stokesI"),
                 frequency=numpy.array([1e8]), channel_bandwidth=numpy.array([1e6]),
                 phasecentre=None) -> Image:
    """Create an empty template image consistent with the inputs.

    :param npixel: Number of pixels
    :param polarisation_frame: Polarisation frame (default PolarisationFrame("stokesI"))
    :param cellsize: cellsize in radians
    :param frequency:
    :param channel_bandwidth: Channel width (Hz)
    :param phasecentre: phasecentre (SkyCoord)
    :return: Image

    """
    
    if phasecentre is None:
        phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
    
    if polarisation_frame is None:
        polarisation_frame = PolarisationFrame("stokesI")
    
    npol = polarisation_frame.npol
    nchan = len(frequency)
    shape = [nchan, npol, npixel, npixel]
    w = WCS(naxis=4)
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, 1.0, channel_bandwidth[0]]
    w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 1.0, frequency[0]]
    w.naxis = 4
    w.wcs.radesys = 'ICRS'
    w.wcs.equinox = 2000.0
    
    return create_image_from_array(numpy.zeros(shape), w, polarisation_frame=polarisation_frame)


def create_image_from_array(data: numpy.array, wcs: WCS, polarisation_frame: PolarisationFrame) -> Image:
    """ Create an image from an array and optional wcs
    
    The output image preserves a reference to the input array.

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
    
    assert isinstance(fim, Image), "Type is %s" % type(fim)
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
    
    assert isinstance(polarisation_frame, PolarisationFrame)
    return polarisation_frame


def checkwcs(wcs1, wcs2):
    """ Check for compatbility of wcs
    
    :param wcs1:
    :param wcs2:
    """
    pass
    # No confidence in this next test
    # assert wcs1.wcs.compare(wcs2.wcs, cmp=1 | 2 | 4), "WCS's do not agree"


def convert_image_to_kernel(im: Image, oversampling, kernelwidth):
    """ Convert an image to a griddata kernel
    
    :param im: Image to be converted
    :param oversampling: Oversampling of Image spatially
    :param kernelwidth: Kernel width to be extracted
    :return: numpy.ndarray[nchan, npol, oversampling, oversampling, kernelwidth, kernelwidth]
    """
    naxis = len(im.shape)
    assert naxis == 4
    
    assert numpy.max(numpy.abs(im.data)) > 0.0, "Image is empty"
    
    nchan, npol, ny, nx = im.shape
    assert nx % oversampling == 0, "Oversampling must be even"
    assert ny % oversampling == 0, "Oversampling must be even"
    
    assert kernelwidth < nx and kernelwidth < ny, "Specified kernel width %d too large"
    
    assert im.wcs.wcs.ctype[0] == 'UU', 'Axis type %s inappropriate for construction of kernel' % im.wcs.wcs.ctype[0]
    assert im.wcs.wcs.ctype[1] == 'VV', 'Axis type %s inappropriate for construction of kernel' % im.wcs.wcs.ctype[1]
    newwcs = WCS(naxis=naxis + 2)
    for axis in range(2):
        newwcs.wcs.ctype[axis] = im.wcs.wcs.ctype[axis]
        newwcs.wcs.crpix[axis] = kernelwidth // 2
        newwcs.wcs.crval[axis] = 0.0
        newwcs.wcs.cdelt[axis] = im.wcs.wcs.cdelt[axis] * oversampling
        
        newwcs.wcs.ctype[axis + 2] = im.wcs.wcs.ctype[axis]
        newwcs.wcs.crpix[axis + 2] = oversampling // 2
        newwcs.wcs.crval[axis + 2] = 0.0
        newwcs.wcs.cdelt[axis + 2] = im.wcs.wcs.cdelt[axis]
        
        # Now do Stokes and Frequency
        newwcs.wcs.ctype[axis + 4] = im.wcs.wcs.ctype[axis + 2]
        newwcs.wcs.crpix[axis + 4] = im.wcs.wcs.crpix[axis + 2]
        newwcs.wcs.crval[axis + 4] = im.wcs.wcs.crval[axis + 2]
        newwcs.wcs.cdelt[axis + 4] = im.wcs.wcs.cdelt[axis + 2]
    
    newdata_shape = [nchan, npol, oversampling, oversampling, kernelwidth, kernelwidth]

    newdata = numpy.zeros(newdata_shape, dtype=im.data.dtype)
    
    assert oversampling * kernelwidth < ny
    assert oversampling * kernelwidth < nx
    
    ystart = ny // 2 - oversampling * kernelwidth // 2
    xstart = nx // 2 - oversampling * kernelwidth // 2
    yend = ny // 2 + oversampling * kernelwidth // 2
    xend = nx // 2 + oversampling * kernelwidth // 2
    for chan in range(nchan):
        for pol in range(npol):
            for y in range(oversampling):
                slicey = slice(yend + y, ystart + y, -oversampling)
                for x in range(oversampling):
                    slicex = slice(xend + x, xstart + x, -oversampling)
                    newdata[chan, pol, y, x, ...] = im.data[chan, pol, slicey, slicex]
    
    return create_image_from_array(newdata, newwcs, polarisation_frame=im.polarisation_frame)


def copy_image(im: Image) -> Image:
    """ Create an image from an array
    
    Performs deepcopy of data_models, breaking reference semantics

    :param im:
    :return: Image
    
    """
    assert isinstance(im, Image), im
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
    assert isinstance(im, Image), im
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
    assert isinstance(fim, Image), "Type is %s" % type(fim)
    return fim


def fft_image(im, template_image=None):
    """ FFT an image, transform WCS as well
    
    Prefer to use axes 'UU---SIN' and 'VV---SIN' but astropy will not accept.
    
    :param im:
    :param template_image:
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
        return create_image_from_array(ft_data, wcs=ft_wcs, polarisation_frame=im.polarisation_frame)
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
        return create_image_from_array(ft_data, wcs=ft_wcs, polarisation_frame=im.polarisation_frame)
    else:
        raise NotImplementedError("Cannot FFT specified axes")


def pad_image(im: Image, shape):
    """Pad an image to desired shape
    
    The wcs crpix is adjusted appropriately
    
    :param im:
    :param shape:
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
        newdata[..., ystart:yend, xstart:xend] = im.data[...]
        return create_image_from_array(newdata, newwcs, polarisation_frame=im.polarisation_frame)


def create_w_term_like(im: Image, w, phasecentre=None, remove_shift=False, dopol=False) -> Image:
    """Create an image with a w term phase term in it:
    
    .. math::

    I(l,m) = e^{-2 \\pi j (w(\\sqrt{1-l^2-m^2}-1)}

    
    The vis phasecentre is used as the delay centre for the w term (i.e. where n==0)

    :param phasecentre:
    :param im: template image
    :param w: w value to evaluate (default is median abs)
    :param remove_shift:
    :param dopol: Do screen in polarisation?
    :return: Image
    """
    
    fim = copy_image(im)
    fim_shape = list(im.shape)
    if not dopol:
        fim_shape[1] = 1
    
    fim_array = numpy.zeros(fim_shape, dtype='complex')
    fim = create_image_from_array(fim_array, wcs=im.wcs, polarisation_frame=im.polarisation_frame)
    
    cellsize = abs(fim.wcs.wcs.cdelt[0]) * numpy.pi / 180.0
    nchan, npol, _, npixel = fim_shape
    if phasecentre is SkyCoord:
        wcentre = phasecentre.to_pixel(im.wcs, origin=0)
    else:
        wcentre = [im.wcs.wcs.crpix[0] - 1.0, im.wcs.wcs.crpix[1] - 1.0]
    
    for chan in range(nchan):
        for pol in range(npol):
            fim.data[chan, pol, ...] = w_beam(npixel, npixel * cellsize, w=w, cx=wcentre[0],
                                              cy=wcentre[1], remove_shift=remove_shift)
    fov = npixel * cellsize
    fresnel = numpy.abs(w) * (0.5 * fov) ** 2
    log.debug('create_w_term_image: For w = %.1f, field of view = %.6f, Fresnel number = %.2f' % (w, fov, fresnel))
    
    return fim
