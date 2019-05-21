""" Functions for ionospheric modelling: see SDP memo 97

"""

import logging

import numpy
from scipy.interpolate import RectBivariateSpline

from data_models.memory_data_models import BlockVisibility
from processing_components.calibration.operations import create_gaintable_from_blockvisibility
from processing_components.visibility.base import create_visibility_from_rows
from processing_components.visibility.iterators import vis_timeslice_iter
from processing_library.util.coordinate_support import hadec_to_azel

log = logging.getLogger(__name__)


def create_gaintable_from_pointingtable(vis, sc, pt, vp, vis_slices=None, scale=1.0, order=3, **kwargs):
    """ Create gaintables from a pointing table

    :param vis:
    :param sc: Sky components for which pierce points are needed
    :param pt: Pointing table
    :param vp: Voltage pattern in AZELGEO frame
    :param scale: Multiply the screen by this factor
    :param order: order of spline (default is 3)
    :return:
    """
    assert isinstance(vis, BlockVisibility)
    assert vp.wcs.wcs.ctype[0] == 'AZELGEO long', vp.wcs.wcs.ctype[0]
    assert vp.wcs.wcs.ctype[1] == 'AZELGEO lati', vp.wcs.wcs.ctype[1]
    
    assert vis.configuration.mount[0] == 'azel', "Mount %s not supported yet" % vis.configuration.mount[0]

    nant = vis.vis.shape[1]
    gaintables = [create_gaintable_from_blockvisibility(vis, **kwargs) for i in sc]
    
    nchan, npol, ny, nx = vp.data.shape
    
    real_spline = RectBivariateSpline(range(ny), range(nx), vp.data[0, 0, ...].real, kx=order, ky=order)
    imag_spline = RectBivariateSpline(range(ny), range(nx), vp.data[0, 0, ...].imag, kx=order, ky=order)
    
    # The time in the Visibility is hour angle in seconds!
    number_bad = 0
    number_good = 0
    
    latitude = vis.configuration.location.lat.rad

    r2d = 180.0 / numpy.pi
    s2r = numpy.pi / 43200.0
    # For each hourangle, we need to calculate the location of a component
    # in AZELGEO. With that we can then look up the relevant gain from the
    # voltage pattern
    for iha, rows in enumerate(vis_timeslice_iter(vis, vis_slices=vis_slices)):
        v = create_visibility_from_rows(vis, rows)
        ha = numpy.average(v.time)
        pt_rows = (pt.time == ha)
        pointing_ha = pt.pointing[pt_rows]
        har = s2r * ha
        
        # Calculate the az el for this hourangle and the phasecentre declination
        azimuth_centre, elevation_centre = hadec_to_azel(har,
                                                         vis.phasecentre.dec.rad,
                                                         latitude)
        
        for icomp, comp in enumerate(sc):
            antgain = numpy.zeros([nant], dtype='complex')
            # Calculate the location of the component in AZELGEO, then add the pointing offset
            # for each antenna
            azimuth_comp, elevation_comp = hadec_to_azel(comp.direction.ra.rad - vis.phasecentre.ra.rad + har,
                                                         comp.direction.dec.rad,
                                                         latitude)
            azimuth_diff = azimuth_comp - azimuth_centre
            elevation_diff = elevation_comp - elevation_centre
            for ant in range(nant):
    
                worldloc = [float((azimuth_diff + pointing_ha[0, ant, 0, 0, 0])*r2d),
                            float((elevation_diff + pointing_ha[0, ant, 0, 0, 1])*r2d),
                            vp.wcs.wcs.crval[2], vp.wcs.wcs.crval[3]]
                try:
                    pixloc = vp.wcs.sub(2).wcs_world2pix([worldloc[:2]], 1)[0]
                    assert pixloc[0] > 2
                    assert pixloc[0] < nx - 3
                    assert pixloc[1] > 2
                    assert pixloc[1] < ny - 3
                    gain = real_spline.ev(pixloc[1], pixloc[0]) + 1j * imag_spline(pixloc[1], pixloc[0])
                    antgain[ant] = 1.0 / (scale * gain)
                    number_good += 1
                except:
                    number_bad += 1
                    antgain[ant] = 0.0
            
            gaintables[icomp].gain[iha, :, :, :] = antgain[:, numpy.newaxis, numpy.newaxis, numpy.newaxis]
            gaintables[icomp].phasecentre = comp.direction
            
        
    if number_bad > 0:
        log.warning(
            "create_gaintable_from_pointingtable: %d points are inside the voltage pattern image" % (number_good))
        log.warning(
            "create_gaintable_from_pointingtable: %d points are outside the voltage pattern image" % (number_bad))

    return gaintables
