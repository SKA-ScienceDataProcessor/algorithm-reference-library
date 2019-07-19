""" Functions for ionospheric modelling: see SDP memo 97

"""

import logging

import numpy
from scipy.interpolate import RectBivariateSpline

from data_models.memory_data_models import BlockVisibility
from processing_components.calibration.operations import create_gaintable_from_blockvisibility
from processing_components.visibility.base import create_visibility_from_rows
from processing_components.visibility.iterators import vis_timeslice_iter
from processing_library.util.coordinate_support import hadec_to_azel, azel_to_hadec

log = logging.getLogger(__name__)

# from SKA document, SKA1 SYSTEM BASELINEDESIGN, 2013


def calculate_noise(bandwidth, int_time):
    print ('calc noise')
    k_b = 1.38064852e-23
    T_sys = 20
    area = numpy.pi*(15/2.)**2 ####??
    eta = 0.78 #####????
    bandwidth = bandwidth
    int_time = int_time
    rms_noise =( numpy.sqrt(2) * k_b * T_sys ) / (area * eta * (numpy.sqrt(bandwidth * int_time)))
    rms_noise*=1e26
    print ('rms_noise', rms_noise)
    return rms_noise



def create_gaintable_from_noise_sources(vis, sc, pt, vp, vis_slices=None, scale=1.0, order=3, **kwargs):
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
        azimuth_centre, elevation_centre = hadec_to_azel(har, vis.phasecentre.dec.rad, latitude)

        for icomp, comp in enumerate(sc):
            antgain = numpy.zeros([nant], dtype='complex')
            # Calculate the location of the component in AZELGEO, then add the pointing offset
            # for each antenna
            hacomp = comp.direction.ra.rad - vis.phasecentre.ra.rad + har
            deccomp = comp.direction.dec.rad
            azimuth_comp, elevation_comp = hadec_to_azel(hacomp, deccomp, latitude)
            
            for ant in range(nant):
    
                wcs_azel = vp.wcs.deepcopy()
    
                az_comp = (azimuth_centre + pointing_ha[0, ant, 0, 0, 0])*r2d
                el_comp = (elevation_centre + pointing_ha[0, ant, 0, 0, 1])*r2d
                
                # We use WCS sensible coordinate handling by labelling the axes misleadingly
                wcs_azel.wcs.crval[0] = az_comp
                wcs_azel.wcs.crval[1] = el_comp
                wcs_azel.wcs.ctype[0] = 'RA---SIN'
                wcs_azel.wcs.ctype[1] = 'DEC--SIN'

                worldloc = [azimuth_comp*r2d, elevation_comp*r2d,
                            vp.wcs.wcs.crval[2], vp.wcs.wcs.crval[3]]
                try:
                    pixloc = wcs_azel.sub(2).wcs_world2pix([worldloc[:2]], 1)[0]
                    assert pixloc[0] > 2
                    assert pixloc[0] < nx - 3
                    assert pixloc[1] > 2
                    assert pixloc[1] < ny - 3
                    rms = calculate_noise(1.e6, 1.)
                    print ('rms', rms)
                    antgain[ant] = numpy.random.normal(0, rms)#1.0 / (scale * gain)
                    print ('antgain ant', antgain[ant])
                    number_good += 1
                except:
                    number_bad += 1
                    antgain[ant] = 0.0
           
            
            gaintables[icomp].gain[iha, :, :, :] = numpy.exp(1j * antgain[:, numpy.newaxis, numpy.newaxis, numpy.newaxis])
         #   gaintables[icomp].gain[iha, :, :, :] = antgain[:, numpy.newaxis, numpy.newaxis, numpy.newaxis]
            print ('gaintables',gaintables[icomp].gain[iha, :, :, :])
            gaintables[icomp].phasecentre = comp.direction
        
    if number_bad > 0:
        log.warning(
            "create_gaintable_from_pointingtable: %d points are inside the voltage pattern image" % (number_good))
        log.warning(
            "create_gaintable_from_pointingtable: %d points are outside the voltage pattern image" % (number_bad))

    return gaintables
