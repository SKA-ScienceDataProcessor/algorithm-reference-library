"""Function to manage sky components.

"""

import collections
import logging
from typing import Union, List

import astropy.units as u
import numpy
from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from photutils import segmentation

from arl.data.data_models import Image, Skycomponent, assert_same_chan_pol
from arl.data.polarisation import PolarisationFrame

log = logging.getLogger(__name__)

def create_skycomponent(direction: SkyCoord, flux: numpy.array, frequency: numpy.array, shape: str = 'Point',
                        polarisation_frame=PolarisationFrame("stokesIQUV"), param: dict = None, name: str = '') \
        -> Skycomponent:
    """ A single Skycomponent with direction, flux, shape, and params for the shape

    :param param:
    :param direction:
    :param flux:
    :param frequency:
    :param shape: 'Point' or 'Gaussian'
    :param name:
    :return: Skycomponent
    """
    return Skycomponent(
        direction=direction,
        frequency=frequency,
        name=name,
        flux=numpy.array(flux),
        shape=shape,
        params=param,
        polarisation_frame=polarisation_frame)


def find_nearest_component(home, comps) -> Skycomponent:
    """ Find nearest component to a given direction
    
    :param home: Home direction
    :param comps: list of skycomponents
    :return: nearest component
    """
    sep = 2 * numpy.pi
    best = None
    for comp in comps:
        thissep = comp.direction.separation(home).rad
        if thissep < sep:
            sep = thissep
            best = comp
    return best


def find_skycomponents(im: Image, fwhm=1.0, threshold=10.0, npixels=5) -> List[Skycomponent]:
    """ Find gaussian components in Image above a certain threshold as Skycomponent

    :param fwhm: Full width half maximum of gaussian
    :param threshold: Threshold for component detection. Default: 10 standard deviations over median.
    :param im: Image to be searched
    :param params:
    :return: list of sky components
    """
    
    assert isinstance(im, Image)
    log.info("find_skycomponents: Finding components in Image by segmentation")
    
    # We use photutils segmentation - this first segments the image
    # into pieces that are thought to contain individual sources, then
    # identifies the concrete source properties. Having these two
    # steps makes it straightforward to extract polarisation and
    # spectral information.
    
    # Make filter kernel
    sigma = fwhm * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma, x_size=int(1.5 * fwhm), y_size=int(1.5 * fwhm))
    kernel.normalize()
    
    # Segment the average over all channels of Stokes I
    image_sum = numpy.sum(im.data, axis=(0))[0, ...] / float(im.shape[0])
    segments = segmentation.detect_sources(image_sum, threshold, npixels=npixels, filter_kernel=kernel)
    log.info("find_skycomponents: Identified %d segments" % segments.nlabels)
    
    # Now get source properties for all polarisations and frequencies
    comp_tbl = [[segmentation.source_properties(im.data[chan, pol], segments,
                                                filter_kernel=kernel, wcs=im.wcs)
                 for pol in [0]]
                for chan in range(im.nchan)]
    
    def comp_prop(comp, prop_name):
        return [[comp_tbl[chan][pol][comp][prop_name]
                 for pol in [0]]
                for chan in range(im.nchan)]
    
    # Generate components
    comps = []
    for segment in range(segments.nlabels):
        # Get flux and position. Astropy's quantities make this
        # unnecessarily complicated.
        flux = numpy.array(comp_prop(segment, "max_value"))
        # These values seem inconsistent with the xcentroid, and ycentroid values
        # ras = u.Quantity(list(map(u.Quantity,
        #         comp_prop(segment, "ra_icrs_centroid"))))
        # decs = u.Quantity(list(map(u.Quantity,
        #         comp_prop(segment, "dec_icrs_centroid"))))
        xs = u.Quantity(list(map(u.Quantity,
                                 comp_prop(segment, "xcentroid"))))
        ys = u.Quantity(list(map(u.Quantity,
                                 comp_prop(segment, "ycentroid"))))
        
        sc = pixel_to_skycoord(xs, ys, im.wcs, 0)
        ras = sc.ra
        decs = sc.dec
        
        # Remove NaNs from RA/DEC (happens if there is no flux in that
        # polarsiation/channel)
        # ras[numpy.isnan(ras)] = 0.0
        # decs[numpy.isnan(decs)] = 0.0
        
        # Determine "true" position by weighting
        flux_sum = numpy.sum(flux)
        ra = numpy.sum(flux * ras) / flux_sum
        dec = numpy.sum(flux * decs) / flux_sum
        xs = numpy.sum(flux * xs) / flux_sum
        ys = numpy.sum(flux * ys) / flux_sum
        
        point_flux = im.data[:, :, numpy.round(ys.value).astype('int'),
                     numpy.round(xs.value).astype('int')]
        
        # Add component
        comps.append(Skycomponent(
            direction=SkyCoord(ra=ra, dec=dec),
            frequency=im.frequency,
            name="Segment %d" % segment,
            flux=point_flux,
            shape='Point',
            polarisation_frame=im.polarisation_frame,
            params={'xpixel': xs, 'ypixel': ys, 'sum_flux': flux}))  # Table has lots of data, could add more in future
    
    return comps


def apply_beam_to_skycomponent(sc: Union[Skycomponent, List[Skycomponent]], beam: Image, flux_limit=0.0) \
        -> Union[Skycomponent, List[Skycomponent]]:
    """ Insert a Skycomponent into an image
    
    :param beam:
    :param sc: SkyComponent or list of SkyComponents
    :param flux_limit: flux limit on input
    :return: List of skycomponents
    """
    assert isinstance(beam, Image)
    single = not isinstance(sc, collections.Iterable)
    
    if single:
        sc = [sc]
    
    nchan, npol, ny, nx = beam.shape
    
    log.debug('apply_beam_to_skycomponent: Processing %d components' % (len(sc)))
    
    newsc = []
    total_flux = numpy.zeros([nchan, npol])
    for comp in sc:
        
        assert comp.shape == 'Point', "Cannot handle shape %s" % comp.shape
        
        assert_same_chan_pol(beam, comp)
        
        pixloc = skycoord_to_pixel(comp.direction, beam.wcs, 1, 'wcs')
        if not numpy.isnan(pixloc).any():
            x, y = int(round(float(pixloc[0]))), int(round(float(pixloc[1])))
            if x >= 0 and x < nx and y >= 0 and y < ny:
                comp.flux[:, :] *= beam.data[:, :, y, x]
                #                if comp.flux[:, :].any() > flux_limit:
                if comp.flux[0, 0] > flux_limit:
                    total_flux += comp.flux
                    newsc.append(Skycomponent(comp.direction, comp.frequency, comp.name, comp.flux,
                                              shape=comp.shape,
                                              polarisation_frame=comp.polarisation_frame))
    
    log.debug('apply_beam_to_skycomponent: %d components with total flux %s' %
              (len(newsc), total_flux))
    if single:
        return newsc[0]
    else:
        return newsc


def insert_skycomponent(im: Image, sc: Union[Skycomponent, List[Skycomponent]], insert_method='Nearest',
                        bandwidth=1.0, support=8) -> Image:
    """ Insert a Skycomponent into an image
    
    :param params:
    :param im:
    :param sc: SkyComponent or list of SkyComponents
    :param insert_method: '' | 'Sinc' | 'Lanczos'
    :param bandwidth: Fractional of uv plane to optimise over (1.0)
    :param support: Support of kernel (7)
    :return: image
    """
    
    assert isinstance(im, Image)
    
    support = int(support / bandwidth)
    
    nchan, npol, ny, nx = im.data.shape
    
    if not isinstance(sc, collections.Iterable):
        sc = [sc]
    
    log.debug("insert_skycomponent: Using insert method %s" % insert_method)
    
    for comp in sc:
        
        assert comp.shape == 'Point', "Cannot handle shape %s" % comp.shape
        
        assert_same_chan_pol(im, comp)
        pixloc = skycoord_to_pixel(comp.direction, im.wcs, origin=0, mode='wcs')
        if insert_method == "Lanczos":
            insert_array(im.data, pixloc[0], pixloc[1], comp.flux, bandwidth, support,
                         insert_function=insert_function_L)
        elif insert_method == "Sinc":
            insert_array(im.data, pixloc[0], pixloc[1], comp.flux, bandwidth, support,
                         insert_function=insert_function_sinc)
        elif insert_method == "PSWF":
            insert_array(im.data, pixloc[0], pixloc[1], comp.flux, bandwidth, support,
                         insert_function=insert_function_pswf)
        else:
            insert_method = 'Nearest'
            y, x = numpy.round(pixloc[1]).astype('int'), numpy.round(pixloc[0]).astype('int')
            if x >= 0 and x < nx and y >= 0 and y < ny:
                im.data[:, :, y, x] += comp.flux
    
    return im


def insert_function_sinc(x):
    s = numpy.zeros_like(x)
    s[x != 0.0] = numpy.sin(numpy.pi * x[x != 0.0]) / (numpy.pi * x[x != 0.0])
    return s


def insert_function_L(x, a=5):
    L = insert_function_sinc(x) * insert_function_sinc(x / a)
    return L


def insert_function_pswf(x, a=5):
    from arl.fourier_transforms.convolutional_gridding import grdsf
    return grdsf(abs(x) / a)[1]


def insert_array(im, x, y, flux, bandwidth=1.0, support=7, insert_function=insert_function_L):
    """ Insert point into image using specified function
    
    :param im: Image
    :param x: x in float pixels
    :param y: y in float pixels
    :param flux: Flux[nchan, npol]
    :param bandwidth: Support of data in uv plane
    :param support: Support of function in image space
    :param insert_function: insert_function_L or insert_function_Sinc or insert_function_pswf
    :return:
    """
    nchan, npol, ny, nx = im.shape
    intx = int(numpy.round(x))
    inty = int(numpy.round(y))
    fracx = x - intx
    fracy = y - inty
    gridx = numpy.arange(-support, support)
    gridy = numpy.arange(-support, support)
    
    insert = numpy.outer(insert_function(bandwidth * (gridy - fracy)),
                         insert_function(bandwidth * (gridx - fracx)))
    
    insertsum = numpy.sum(insert)
    assert insertsum > 0, "Sum of interpolation coefficients %g" % insertsum
    insert = insert / insertsum
    
    for chan in range(nchan):
        for pol in range(npol):
            im[chan, pol, inty - support:inty + support, intx - support:intx + support] += flux[chan, pol] * insert
    
    return im
