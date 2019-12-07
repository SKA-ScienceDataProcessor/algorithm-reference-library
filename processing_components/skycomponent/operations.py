"""Function to manage sky components.

"""

__all__ = ['create_skycomponent', 'filter_skycomponents_by_flux', 'filter_skycomponents_by_index',
           'find_nearest_skycomponent', 'find_nearest_skycomponent_index', 'find_separation_skycomponents',
           'find_skycomponent_matches', 'find_skycomponent_matches_atomic', 'find_skycomponents',
           'insert_skycomponent', 'voronoi_decomposition', 'image_voronoi_iter',
           'partition_skycomponent_neighbours', 'select_components_by_separation', 'select_neighbouring_components',
           'remove_neighbouring_components', 'apply_beam_to_skycomponent']

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
from scipy.spatial import Voronoi

from data_models.memory_data_models import Image, Skycomponent, assert_same_chan_pol
from data_models.polarisation import PolarisationFrame
from processing_library.image.operations import create_image_from_array
from processing_library.util.array_functions import insert_function_sinc, insert_function_L, \
    insert_function_pswf, insert_array

log = logging.getLogger(__name__)


def create_skycomponent(direction: SkyCoord, flux: numpy.array, frequency: numpy.array, shape: str = 'Point',
                        polarisation_frame=PolarisationFrame("stokesIQUV"), params: dict = None, name: str = '') \
        -> Skycomponent:
    """ A single Skycomponent with direction, flux, shape, and params for the shape

    :param polarisation_frame:
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

    :param tol:
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


def select_components_by_separation(home, comps, rmax=2 * numpy.pi, rmin=0.0) -> [Skycomponent]:
    """ Select components with a range in separation

    :param home: Home direction
    :param comps: list of skycomponents
    :param rmin: minimum range
    :param rmax: maximum range
    :return: selected components
    """
    selected = list()
    for comp in comps:
        thissep = comp.direction.separation(home).rad
        if rmin <= thissep <= rmax:
            selected.append(comp)
    return selected



def select_neighbouring_components(comps, target_comps):
    """ Assign components to nearest in the target
    
    :param comps:
    :param target_comps:
    :return: Indices of components in target_comps
    """
    target_catalog = SkyCoord([c.direction.ra.rad for c in target_comps] * u.rad,
                              [c.direction.dec.rad for c in target_comps] * u.rad)
    
    all_catalog = SkyCoord([c.direction.ra.rad for c in comps] * u.rad,
                           [c.direction.dec.rad for c in comps] * u.rad)
    
    from astropy.coordinates import match_coordinates_sky
    idx, d2d, d3d = match_coordinates_sky(all_catalog, target_catalog)
    return idx, d2d


def remove_neighbouring_components(comps, distance):
    """ Remove the faintest of a pair of components that are within a specified distance

    :param comps:
    :param target_comps:
    :param distance: Minimum distance
    :return: Indices of components in target_comps, selected components
    """
    ncomps = len(comps)
    ok = ncomps * [True]
    for i in range(ncomps):
        if ok[i]:
            for j in range(i+1, ncomps):
                if ok[j]:
                    d = comps[i].direction.separation(comps[j].direction).rad
                    if d < distance:
                        if numpy.max(comps[i].flux) > numpy.max(comps[j].flux):
                            ok[j] = False
                        else:
                            ok[i] = False
                        break

    from itertools import compress
    idx = list(compress(list(range(ncomps)), ok))
    comps_sel = list(compress(comps, ok))
    return idx, comps_sel


def find_skycomponents(im: Image, fwhm=1.0, threshold=1.0, npixels=5) -> List[Skycomponent]:
    """ Find gaussian components in Image above a certain threshold as Skycomponent

    :param im: Image to be searched
    :param fwhm: Full width half maximum of gaussian in pixels
    :param threshold: Threshold for component detection. Default: 1 Jy.
    :param npixels: Number of connected pixels required
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
    image_sum = numpy.sum(im.data, axis=0)[0, ...] / float(im.shape[0])
    segments = segmentation.detect_sources(image_sum, threshold, npixels=npixels, filter_kernel=kernel)
    log.info("find_skycomponents: Identified %d segments" % segments.nlabels)
    
    # Now compute source properties for all polarisations and frequencies
    comp_tbl = [[segmentation.source_properties(im.data[chan, pol], segments,
                                                filter_kernel=kernel,
                                                wcs=im.wcs.sub([1,2])).to_table()
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
            if 0 <= x < nx and 0 <= y < ny:
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
            if 0 <= x < nx and 0 <= y < ny:
                im.data[:, :, y, x] += flux[...]
    
    return im


def voronoi_decomposition(im, comps):
    """Construct a Voronoi decomposition of a set of components

    The array return contains the index into the Voronoi structure

    :param im:
    :param comps:
    :return: Voronoi structure, vertex image
    """
    
    def voronoi_vertex(vy, vx, vertex_y, vertex_x):
        """ Return the nearest Voronoi vertex

        :param vy:
        :param vx:
        :param vertex_y:
        :param vertex_x:
        :return:
        """
        return numpy.argmin(numpy.hypot(vy - vertex_y, vx - vertex_x))
    
    directions = SkyCoord([u.rad * c.direction.ra.rad for c in comps],
                          [u.rad * c.direction.dec.rad for c in comps])
    x, y = skycoord_to_pixel(directions, im.wcs, 0, 'wcs')
    points = [(x[i], y[i]) for i, _ in enumerate(x)]
    vor = Voronoi(points)
    
    nchan, npol, ny, nx = im.shape
    vertex_image = numpy.zeros([ny, nx]).astype('int')
    for j in range(ny):
        for i in range(nx):
            vertex_image[j, i] = voronoi_vertex(j, i, vor.points[:, 1], vor.points[:, 0])
    
    return vor, vertex_image


def image_voronoi_iter(im: Image, components: Skycomponent) -> collections.Iterable:
    """Iterate through Voronoi decomposition, returning a generator yielding fullsize images

    :param im: Image
    :param components: Components to define Voronoi decomposition
    """
    if len(components) == 1:
        mask = numpy.ones(im.data.shape)
        yield create_image_from_array(mask, wcs=im.wcs,
                                      polarisation_frame=im.polarisation_frame)
    else:
        vor, vertex_array = voronoi_decomposition(im, components)
        
        nregions = numpy.max(vertex_array) + 1
        for region in range(nregions):
            mask = numpy.zeros(im.data.shape)
            mask[(vertex_array == region)[numpy.newaxis, numpy.newaxis,...]] = 1.0
            yield create_image_from_array(mask, wcs=im.wcs,
                                          polarisation_frame=im.polarisation_frame)

def partition_skycomponent_neighbours(comps, targets):
    """ Partition sky components by nearest target source
    :param comps:
    :param targets:
    :return:
    """
    idx, d2d = select_neighbouring_components(comps, targets)

    from itertools import compress
    comps_lists = list()
    for comp_id in numpy.unique(idx):
        selected_comps = list(compress(comps, idx == comp_id))
        comps_lists.append(selected_comps)
        
    return comps_lists