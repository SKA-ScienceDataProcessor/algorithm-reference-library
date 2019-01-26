""" Functions for ionospheric modelling: see SDP memo 97

"""

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.memory_data_models import BlockVisibility
from processing_components.calibration.operations import create_gaintable_from_blockvisibility, \
    create_gaintable_from_rows
from processing_components.calibration.iterators import gaintable_timeslice_iter
from processing_components.image.operations import copy_image, create_empty_image_like
from processing_components.visibility.base import create_visibility_from_rows
from processing_components.visibility.iterators import vis_timeslice_iter
from processing_library.util.coordinate_support import xyz_to_uvw, skycoord_to_lmn

import logging
log = logging.getLogger(__name__)

def find_pierce_points(station_locations, ha, dec, phasecentre, height):
    """Find the pierce points for a flat screen at specified height
    
    :param station_locations: All station locations [:3]
    :param ha: Hour angle
    :param dec: Declination
    :param phasecentre: Phase centre
    :param height: Height of screen
    :return:
    """
    source_direction = SkyCoord(ra=ha, dec=dec, frame='icrs', equinox='J2000')
    local_locations = xyz_to_uvw(station_locations, ha, dec)
    local_locations -= numpy.average(local_locations, axis=0)
    
    lmn = numpy.array(skycoord_to_lmn(source_direction, phasecentre))
    lmn[2] += 1.0
    pierce_points = local_locations + height * numpy.array(lmn)
    return pierce_points


def create_gaintable_from_screen(vis, sc, screen, height=3e5, vis_slices=None, scale=1.0, **kwargs):
    """ Create gaintables from a screen calculated using ARatmospy

    :param vis:
    :param sc: Sky components for which pierce points are needed
    :param screen:
    :param height: Height (in m) of screen above telescope e.g. 3e5
    :param scale: Multiply the screen by this factor
    :return:
    """
    assert isinstance(vis, BlockVisibility)
    
    station_locations = vis.configuration.xyz
    
    nant = station_locations.shape[0]
    t2r = numpy.pi / 43200.0
    gaintables = [create_gaintable_from_blockvisibility(vis, **kwargs) for i in sc]
    
    # The time in the Visibility is hour angle in seconds!
    for iha, rows in enumerate(vis_timeslice_iter(vis, vis_slices=vis_slices)):
        v = create_visibility_from_rows(vis, rows)
        ha = numpy.average(v.time)
        number_bad = 0
        number_good = 0

        for icomp, comp in enumerate(sc):
            pp = find_pierce_points(station_locations, (comp.direction.ra.rad + t2r * ha) * u.rad, comp.direction.dec,
                                    height=height, phasecentre=vis.phasecentre)
            scr = numpy.zeros([nant])
            for ant in range(nant):
                pp0 = pp[ant][0:2]
                worldloc = [pp0[0], pp0[1], ha, 1e8]
                try:
                    pixloc = screen.wcs.wcs_world2pix([worldloc], 0)[0].astype('int')
                    scr[ant] = scale * screen.data[pixloc[3], pixloc[2], pixloc[1], pixloc[0]]
                    number_good += 1
                except:
                    number_bad += 1
                    scr[ant] = 0.0
            
            gaintables[icomp].gain[iha, :, :, :] = numpy.exp(1j * scr[:, numpy.newaxis, numpy.newaxis, numpy.newaxis])
            gaintables[icomp].phasecentre = comp.direction
        
        if number_bad > 0:
            log.warning("create_gaintable_from_screen: %d pierce points are inside the screen image" % (number_good))
            log.warning("create_gaintable_from_screen: %d pierce points are outside the screen image" % (number_bad))

    return gaintables


def grid_gaintable_to_screen(vis, gaintables, screen, height=3e5, gaintable_slices=None, scale=1.0, **kwargs):
    """ Grid a gaintable to a screen image
    
    The phases are just average per grid cell, no phase unwrapping is performed.

    :param vis:
    :param sc: Sky components for which pierce points are needed
    :param screen:
    :param height: Height (in m) of screen above telescope e.g. 3e5
    :param scale: Multiply the screen by this factor
    :return: gridded screen image, weights image
    """
    assert isinstance(vis, BlockVisibility)
    
    station_locations = vis.configuration.xyz
    
    nant = station_locations.shape[0]
    t2r = numpy.pi / 43200.0
    
    newscreen = create_empty_image_like(screen)
    weights = create_empty_image_like(screen)
    nchan, ntimes, ny, nx = screen.shape

    # The time in the Visibility is hour angle in seconds!
    number_no_weight = 0
    for gaintable in gaintables:
        for iha, rows in enumerate(gaintable_timeslice_iter(gaintable, gaintable_slices=gaintable_slices)):
            gt = create_gaintable_from_rows(gaintable, rows)
            ha = numpy.average(gt.time)
        
            pp = find_pierce_points(station_locations,
                                    (gt.phasecentre.ra.rad + t2r * ha) * u.rad,
                                    gt.phasecentre.dec,
                                    height=height,
                                    phasecentre=vis.phasecentre)
            scr = numpy.angle(gt.gain[0, :, 0, 0, 0])
            wt = gt.weight[0, :, 0, 0, 0]
            for ant in range(nant):
                pp0 = pp[ant][0:2]
                worldloc = [pp0[0], pp0[1], ha, 1e8]
                pixloc = newscreen.wcs.wcs_world2pix([worldloc], 0)[0].astype('int')
                assert pixloc[0] >= 0
                assert pixloc[0] < nx
                assert pixloc[1] >= 0
                assert pixloc[1] < ny
                newscreen.data[pixloc[3], pixloc[2], pixloc[1], pixloc[0]] += wt[ant] * scr[ant]
                weights.data[pixloc[3], pixloc[2], pixloc[1], pixloc[0]] += wt[ant]
                if wt[ant] == 0.0:
                    number_no_weight += 1
    if number_no_weight > 0:
        print("grid_gaintable_to_screen: %d pierce points are have no weight" % (number_no_weight))
        log.warning("grid_gaintable_to_screen: %d pierce points are have no weight" % (number_no_weight))

    newscreen.data[weights.data > 0.0] = newscreen.data[weights.data > 0.0] / weights.data[weights.data > 0.0]

    return newscreen, weights


def calculate_sf_from_screen(screen):
    """ Calculate structure function image from screen

    Screen axes are ['XX', 'YY', 'TIME', 'FREQ']
    :param screen:
    :return:
    """
    from scipy.signal import fftconvolve
    nchan, ntimes, ny, nx = screen.data.shape
    
    sf = numpy.zeros([nchan, 1, 2 * ny - 1, 2 * nx - 1])
    for chan in range(nchan):
        sf[chan, 0, ...] = fftconvolve(screen.data[chan, 0, ...], screen.data[chan, 0, ::-1, ::-1])
        for itime in range(ntimes):
            sf += fftconvolve(screen.data[chan, itime, ...], screen.data[chan, itime, ::-1, ::-1])
        sf[chan, 0, ...] /= numpy.max(sf[chan, 0, ...])
        sf[chan, 0, ...] = 1.0 - sf[chan, 0, ...]
    
    sf_image = copy_image(screen)
    sf_image.data = sf[:, :, (ny - ny // 4):(ny + ny // 4), (nx - nx // 4):(nx + nx // 4)]
    sf_image.wcs.wcs.crpix[0] = ny // 4 + 1
    sf_image.wcs.wcs.crpix[1] = ny // 4 + 1
    sf_image.wcs.wcs.crpix[2] = 1
    
    return sf_image


def plot_gaintable_on_screen(vis, gaintables, height=3e5, gaintable_slices=None, plotfile=None):
    """ Plot a gaintable on an ionospheric screen

    :param vis:
    :param sc: Sky components for which pierce points are needed
    :param height: Height (in m) of screen above telescope e.g. 3e5
    :param scale: Multiply the screen by this factor
    :return: gridded screen image, weights image
    """
    
    import matplotlib.pyplot as plt
    
    assert isinstance(vis, BlockVisibility)
    
    station_locations = vis.configuration.xyz
    
    t2r = numpy.pi / 43200.0
    
    # The time in the Visibility is hour angle in seconds!
    plt.clf()
    for gaintable in gaintables:
        for iha, rows in enumerate(gaintable_timeslice_iter(gaintable, gaintable_slices=gaintable_slices)):
            gt = create_gaintable_from_rows(gaintable, rows)
            ha = numpy.average(gt.time)
            
            pp = find_pierce_points(station_locations,
                                    (gt.phasecentre.ra.rad + t2r * ha) * u.rad,
                                    gt.phasecentre.dec,
                                    height=height,
                                    phasecentre=vis.phasecentre)
            phases = numpy.angle(gt.gain[0, :, 0, 0, 0])
            plt.scatter(pp[:,0],pp[:,1], c=phases, cmap='hsv', alpha=0.75, s=0.1)
            
    plt.title('Pierce point phases')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    
    if plotfile is not None:
        plt.savefig(plotfile)

    plt.show()