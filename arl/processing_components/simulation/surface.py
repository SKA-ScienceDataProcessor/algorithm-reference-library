""" Functions for dish surface modelling

"""

__all__ = ['simulate_gaintable_from_voltage_patterns']

import logging

import numpy
from scipy.interpolate import RectBivariateSpline

from arl.data_models.memory_data_models import BlockVisibility
from arl.processing_components.calibration.operations import create_gaintable_from_blockvisibility
from arl.processing_components.visibility.base import create_visibility_from_rows
from arl.processing_components.visibility.iterators import vis_timeslice_iter
from arl.processing_library.util.coordinate_support import hadec_to_azel, azel_to_hadec

log = logging.getLogger(__name__)

def simulate_gaintable_from_voltage_patterns(vis, sc, vp_list, vp_coeffs, vis_slices=None, order=3, use_radec=False,
                                             **kwargs):
    """ Create gaintables for a set of zernikes

    :param vis:
    :param sc: Sky components for which pierce points are needed
    :param vp: List of Voltage patterns in AZELGEO frame
    :param vp_coeffs: Fractional contribution [nants, nvp]
    :param order: order of spline (default is 3)
    :return:
    """

    ntimes, nant = vis.vis.shape[0:2]
    vp_coeffs = numpy.array(vp_coeffs)
    gaintables = [create_gaintable_from_blockvisibility(vis, **kwargs) for i in sc]

    if not use_radec:
        assert isinstance(vis, BlockVisibility)
        assert vis.configuration.mount[0] == 'azel', "Mount %s not supported yet" % vis.configuration.mount[0]
    
        # The time in the Visibility is hour angle in seconds!
        number_bad = 0
        number_good = 0

        # Cache the splines, one per voltage pattern
        real_splines = list()
        imag_splines = list()
        for ivp, vp in enumerate(vp_list):
            assert vp.wcs.wcs.ctype[0] == 'AZELGEO long', vp.wcs.wcs.ctype[0]
            assert vp.wcs.wcs.ctype[1] == 'AZELGEO lati', vp.wcs.wcs.ctype[1]
    
            nchan, npol, ny, nx = vp.data.shape
            real_splines.append(RectBivariateSpline(range(ny), range(nx), vp.data[0, 0, ...].real, kx=order,
                                              ky=order))
            imag_splines.append(RectBivariateSpline(range(ny), range(nx), vp.data[0, 0, ...].imag, kx=order,
                                              ky=order))

        latitude = vis.configuration.location.lat.rad
    
        r2d = 180.0 / numpy.pi
        s2r = numpy.pi / 43200.0
        # For each hourangle, we need to calculate the location of a component
        # in AZELGEO. With that we can then look up the relevant gain from the
        # voltage pattern
        for iha, rows in enumerate(vis_timeslice_iter(vis, vis_slices=vis_slices)):
                v = create_visibility_from_rows(vis, rows)
                ha = numpy.average(v.time)
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
                        for ivp, vp in enumerate(vp_list):
                            nchan, npol, ny, nx = vp.data.shape
                            wcs_azel = vp.wcs.deepcopy()
            
                            az_comp = azimuth_centre * r2d
                            el_comp = elevation_centre * r2d
                        
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
                                gain = real_splines[ivp].ev(pixloc[1], pixloc[0]) \
                                       + 1j * imag_splines[ivp](pixloc[1], pixloc[0])
                                antgain[ant] += vp_coeffs[ant, ivp] * gain
                                number_good += 1
                            except:
                                number_bad += 1
                                antgain[ant] = 0.0
                                
                        antgain[ant] = 1.0/antgain[ant]
                    
                    gaintables[icomp].gain[iha, :, :, :] = antgain[:, numpy.newaxis, numpy.newaxis, numpy.newaxis]
                    gaintables[icomp].phasecentre = comp.direction
    else:
        assert isinstance(vis, BlockVisibility)
        number_bad = 0
        number_good = 0

        # Cache the splines, one per voltage pattern
        real_splines = list()
        imag_splines = list()
        for ivp, vp in enumerate(vp_list):
    
            nchan, npol, ny, nx = vp.data.shape
            real_splines.append(RectBivariateSpline(range(ny), range(nx), vp.data[0, 0, ...].real, kx=order,
                                                    ky=order))
            imag_splines.append(RectBivariateSpline(range(ny), range(nx), vp.data[0, 0, ...].imag, kx=order,
                                                    ky=order))

        for iha, rows in enumerate(vis_timeslice_iter(vis, vis_slices=vis_slices)):

            # The time in the Visibility is hour angle in seconds!
            r2d = 180.0 / numpy.pi
            # For each hourangle, we need to calculate the location of a component
            # in AZELGEO. With that we can then look up the relevant gain from the
            # voltage pattern
            v = create_visibility_from_rows(vis, rows)
            ha = numpy.average(v.time)
        
            for icomp, comp in enumerate(sc):
                antgain = numpy.zeros([nant], dtype='complex')
                antwt = numpy.zeros([nant])
                ra_comp = comp.direction.ra.rad
                dec_comp = comp.direction.dec.rad
                for ant in range(nant):
                    for ivp, vp in enumerate(vp_list):
                        
                        assert vp.wcs.wcs.ctype[0] == 'RA---SIN', vp.wcs.wcs.ctype[0]
                        assert vp.wcs.wcs.ctype[1] == 'DEC--SIN', vp.wcs.wcs.ctype[1]
                        
                        worldloc = [ra_comp * r2d, dec_comp * r2d,
                                    vp.wcs.wcs.crval[2], vp.wcs.wcs.crval[3]]
                        nchan, npol, ny, nx = vp.data.shape

                        try:
                            pixloc = vp.wcs.sub(2).wcs_world2pix([worldloc[:2]], 1)[0]
                            assert pixloc[0] > 2
                            assert pixloc[0] < nx - 3
                            assert pixloc[1] > 2
                            assert pixloc[1] < ny - 3
                            gain = real_splines[ivp].ev(pixloc[1], pixloc[0]) \
                                   + 1j * imag_splines[ivp](pixloc[1], pixloc[0])
                            antgain[ant] += vp_coeffs[ant, ivp] * gain
                            antwt[ant] = 1.0
                            number_good += 1
                        except:
                            number_bad += 1
                            antgain[ant] = 1e15
                            antwt[ant] = 0.0

                        antgain[ant] = 1.0/antgain[ant]

                    gaintables[icomp].gain[iha, :, :, :] = antgain[:, numpy.newaxis, numpy.newaxis, numpy.newaxis]
                    gaintables[icomp].weight[iha, :, :, :] = antwt[:, numpy.newaxis, numpy.newaxis, numpy.newaxis]
                    gaintables[icomp].phasecentre = comp.direction

    if number_bad > 0:
        log.warning(
            "simulate_gaintable_from_voltage_patterns: %d points are inside the voltage pattern image" % (number_good))
        log.warning(
            "simulate_gaintable_from_voltage_patterns: %d points are outside the voltage pattern image" % (number_bad))

    return gaintables