import logging

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.memory_data_models import BlockVisibility, SkyModel

log = logging.getLogger()

from processing_library.util.coordinate_support import xyz_to_uvw, skycoord_to_lmn
from processing_components.calibration.operations import create_gaintable_from_blockvisibility
from processing_components.visibility.iterators import vis_timeslice_iter
from processing_components.visibility.base import create_visibility_from_rows
from processing_components.image.operations import copy_image
from processing_components.visibility.operations import copy_visibility
from processing_components.calibration.operations import copy_gaintable


def find_pierce_points(station_locations, ha, dec, phasecentre, height):
    """Find the pierce points for a screen at specified height
    
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
                except:
                    number_bad += 1
                    scr[ant] = 0.0
            
            gaintables[icomp].gain[iha, :, :, :] = numpy.exp(1j * scr[:, numpy.newaxis, numpy.newaxis, numpy.newaxis])
        
        if number_bad > 0:
            log.warning("create_gaintable_from_screen: %d pierce points are outside the screen image" % (number_bad))
    
    return gaintables


def expand_skymodel_by_skycomponents(sm, **kwargs):
    """ Expand a sky model so that all components are in separate skymodels

    """
    return [SkyModel(components=[comp],
                     image=copy_image(sm.image),
                     gaintable=copy_gaintable(sm.gaintable),
                     mask=copy_image(sm.mask),
                     fixed=sm.fixed) for comp in sm.components]


def sum_visibility_over_partitions(blockvis_list):
    """Sum all the visibility partitions
    
    :param blockvis_list:
    :return: Single visibility
    """
    result = copy_visibility(blockvis_list[0])
    for i, v in enumerate(blockvis_list):
        if i > 0:
            result.data['vis'] += v.data['vis']
    
    return result


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
