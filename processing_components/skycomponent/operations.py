"""Function to manage sky components.

"""

import collections
import logging
from typing import Union, List

import astropy.units as u
import numpy
from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs.utils import pixel_to_skycoord
from astropy.wcs.utils import skycoord_to_pixel
from photutils import segmentation
from scipy import interpolate

from data_models.memory_data_models import Image, Skycomponent, assert_same_chan_pol
from data_models.polarisation import PolarisationFrame
from processing_library.util.array_functions import insert_function_sinc, insert_function_L, insert_function_pswf, insert_array

log = logging.getLogger(__name__)


def create_skycomponent(direction: SkyCoord, flux: numpy.array, frequency: numpy.array, shape: str = 'Point',
                        polarisation_frame=PolarisationFrame("stokesIQUV"), params: dict = None, name: str = '') \
        -> Skycomponent:
    """ A single Skycomponent with direction, flux, shape, and params for the shape

    :param params:
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
        params=params,
        polarisation_frame=polarisation_frame)


def find_nearest_skycomponent_index(home, comps) -> int:
    """ Find nearest component in a list to a given direction (home)

    :param home: Home direction
    :param comps: list of skycomponents
    :return: index of best in comps
    """
    catalog = SkyCoord(ra=[c.direction.ra for c in comps], dec=[c.direction.dec for c in comps])
    idx, dist2d, dist3d = match_coordinates_sky(home, catalog)
    return idx


def find_nearest_skycomponent(home: SkyCoord, comps) -> (Skycomponent, float):
    """ Find nearest component to a given direction

    :param home: Home direction
    :param comps: list of skycomponents
    :return: Index of nearest component
    """
    best_index = find_nearest_skycomponent_index(home, comps)
    best = comps[best_index]
    return best, best.direction.separation(home).rad


def find_separation_skycomponents(comps_test, comps_ref=None):
    """ Find the matrix of separations for two lists of components
    
    :param comps_test: List of components to be test
    :param comps_ref: If None then set to comps_test
    :return:
    """
    if comps_ref is None:
        ncomps = len(comps_test)
        distances = numpy.zeros([ncomps, ncomps])
        for i in range(ncomps):
            for j in range(i + 1, ncomps):
                distances[i, j] = comps_test[i].direction.separation(comps_test[j].direction).rad
                distances[j, i] = distances[i, j]
        return distances
    
    else:
        ncomps_ref = len(comps_ref)
        ncomps_test = len(comps_test)
        separations = numpy.zeros([ncomps_ref, ncomps_test])
        for ref in range(ncomps_ref):
            for test in range(ncomps_test):
                separations[ref, test] = comps_test[test].direction.separation(comps_ref[ref].direction).rad
        
        return separations


def find_skycomponent_matches_atomic(comps_test, comps_ref, tol=1e-7):
    """ Match a list of candidates to a reference set of skycomponents
    
    find_skycomponent_matches is faster since it uses the astropy catalog matching

    many to one is allowed.

    :param comps_test:
    :param comps_ref:
    :return:
    """
    separations = find_separation_skycomponents(comps_test, comps_ref)
    matches = []
    for test, comp_test in enumerate(comps_test):
        best = numpy.argmin(separations[:, test])
        best_sep = separations[best, test]
        if best_sep < tol:
            matches.append((test, best, best_sep))
    
    assert len(matches) <= len(comps_test)
    
    return matches


def find_skycomponent_matches(comps_test, comps_ref, tol=1e-7):
    """ Match a list of candidates to a reference set of skycomponents

    many to one is allowed.

    :param comps_test:
    :param comps_ref:
    :param tol: Tolerance in radians for a match
    :return:
    """
    catalog_test = SkyCoord(ra=[c.direction.ra for c in comps_test],
                            dec=[c.direction.dec for c in comps_test])
    catalog_ref = SkyCoord(ra=[c.direction.ra for c in comps_ref],
                           dec=[c.direction.dec for c in comps_ref])
    idx, dist2d, dist3d = match_coordinates_sky(catalog_test, catalog_ref)
    matches = list()
    for test, comp_test in enumerate(comps_test):
        best = idx[test]
        best_sep = dist2d[test].rad
        if best_sep < tol:
            matches.append((test, best, best_sep))
    
    return matches


def select_components_by_separation(home, comps, max=2 * numpy.pi, min=0.0) -> [Skycomponent]:
    """ Select components with a range in separation

    :param home: Home direction
    :param comps: list of skycomponents
    :param min: minimum range
    :param max: maximum range
    :return: selected components
    """
    selected = list()
    for comp in comps:
        thissep = comp.direction.separation(home).rad
        if thissep >= min and thissep <= max:
            selected.append(comp)
    return selected


def select_components_by_flux(comps, fmax=numpy.infty, fmin=-numpy.infty) -> [Skycomponent]:
    """ Select components with a range in flux

    :param comps: list of skycomponents
    :param fmin: minimum range
    :param fmax: maximum range
    :return: selected components
    """
    selected = list()
    for comp in comps:
        flux = numpy.max(comp.flux)
        if flux >= fmin and flux <= fmax:
            selected.append(comp)
    return selected

def select_neighbouring_components(comps, target_comps, **kwargs):
    """ Assign components to nearest in the target
    
    :param comps:
    :param target_comps:
    :param kwargs:
    :return: Indices of components in target_comps
    """
    target_catalog = SkyCoord([c.direction.ra.rad for c in target_comps] * u.rad,
                              [c.direction.dec.rad for c in target_comps] * u.rad)

    all_catalog = SkyCoord([c.direction.ra.rad for c in comps] * u.rad,
                           [c.direction.dec.rad for c in comps] * u.rad)

    from astropy.coordinates import match_coordinates_sky
    idx, d2d, d3d = match_coordinates_sky(all_catalog, target_catalog)
    return idx, d2d

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
    
    # Now compute source properties for all polarisations and frequencies
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
        
        point_flux = im.data[:, :, numpy.round(ys.value).astype('int'), numpy.round(xs.value).astype('int')]
        
        # Add component
        comps.append(Skycomponent(
            direction=SkyCoord(ra=ra, dec=dec),
            frequency=im.frequency,
            name="Segment %d" % segment,
            flux=point_flux,
            shape='Point',
            polarisation_frame=im.polarisation_frame,
            params={}))
    #           params={'xpixel': xs, 'ypixel': ys, 'sum_flux': flux}))  # Table has lots of data_models, could add more in future
    
    return comps


def apply_beam_to_skycomponent(sc: Union[Skycomponent, List[Skycomponent]], beam: Image) \
        -> Union[Skycomponent, List[Skycomponent]]:
    """ Insert a Skycomponent into an image
    
    :param beam:
    :param sc: SkyComponent or list of SkyComponents
    :return: List of skycomponents
    """
    assert isinstance(beam, Image)
    single = not isinstance(sc, collections.Iterable)
    
    if single:
        sc = [sc]
    
    nchan, npol, ny, nx = beam.shape
    
    log.debug('apply_beam_to_skycomponent: Processing %d components' % (len(sc)))
    
    ras = [comp.direction.ra.radian for comp in sc]
    decs = [comp.direction.dec.radian for comp in sc]
    skycoords = SkyCoord(ras * u.rad, decs * u.rad, frame='icrs')
    pixlocs = skycoord_to_pixel(skycoords, beam.wcs, origin=1, mode='wcs')

    
    newsc = []
    total_flux = numpy.zeros([nchan, npol])
    for icomp, comp in enumerate(sc):
        
        assert comp.shape == 'Point', "Cannot handle shape %s" % comp.shape
        
        assert_same_chan_pol(beam, comp)
        
        pixloc = (pixlocs[0][icomp], pixlocs[1][icomp])
        if not numpy.isnan(pixloc).any():
            x, y = int(round(float(pixloc[0]))), int(round(float(pixloc[1])))
            if x >= 0 and x < nx and y >= 0 and y < ny:
                comp.flux[:, :] *= beam.data[:, :, y, x]
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


def filter_skycomponents_by_index(sc, indices):
    """Filter sky components by index

    :param sc:
    :param indices:
    :return:
    """
    newcomps = list()
    for i in indices:
        newcomps.append(sc[i])
    
    return newcomps


def filter_skycomponents_by_flux(sc, flux_min=-numpy.inf, flux_max=numpy.inf):
    """Filter sky components by stokes I flux

    :param sc:
    :param flux_min:
    :param flux_max:
    :return:
    """
    newcomps = list()
    for comp in sc:
        if (numpy.max(comp.flux[:, 0]) > flux_min) and (numpy.max(comp.flux[:, 0]) < flux_max):
            newcomps.append(comp)
    
    return newcomps


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
    
    image_frequency = im.frequency
    
    ras = [comp.direction.ra.radian for comp in sc]
    decs = [comp.direction.dec.radian for comp in sc]
    skycoords = SkyCoord(ras * u.rad, decs * u.rad, frame='icrs')
    pixlocs = skycoord_to_pixel(skycoords, im.wcs, origin=0, mode='wcs')

    for icomp, comp in enumerate(sc):
        
        assert comp.shape == 'Point', "Cannot handle shape %s" % comp.shape
        
        assert_same_chan_pol(im, comp)
        pixloc = (pixlocs[0][icomp], pixlocs[1][icomp])
        flux = numpy.zeros([nchan, npol])
        
        if comp.flux.shape[0] > 1:
            for pol in range(npol):
                fint = interpolate.interp1d(comp.frequency, comp.flux[:, pol], kind="cubic")
                flux[:, pol] = fint(image_frequency)
        else:
            flux = comp.flux
        
        if insert_method == "Lanczos":
            insert_array(im.data, pixloc[0], pixloc[1], flux, bandwidth, support,
                         insert_function=insert_function_L)
        elif insert_method == "Sinc":
            insert_array(im.data, pixloc[0], pixloc[1], flux, bandwidth, support,
                         insert_function=insert_function_sinc)
        elif insert_method == "PSWF":
            insert_array(im.data, pixloc[0], pixloc[1], flux, bandwidth, support,
                         insert_function=insert_function_pswf)
        else:
            insert_method = 'Nearest'
            y, x = numpy.round(pixloc[1]).astype('int'), numpy.round(pixloc[0]).astype('int')
            if x >= 0 and x < nx and y >= 0 and y < ny:
                im.data[:, :, y, x] += flux[...]
    
    return im
