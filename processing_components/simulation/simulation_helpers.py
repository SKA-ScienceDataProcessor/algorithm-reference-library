""" Functions that help with SKA simulations

"""

import logging

import astropy.units as units
import matplotlib.pyplot as plt
import numpy
from astropy.coordinates import SkyCoord

from data_models.memory_data_models import Skycomponent
from data_models.polarisation import PolarisationFrame
from processing_library.image.operations import create_image
from processing_library.util.coordinate_support import hadec_to_azel
from wrappers.serial.image.operations import show_image
from wrappers.serial.imaging.primary_beams import create_pb
from wrappers.serial.skycomponent.base import copy_skycomponent
from wrappers.serial.skycomponent.operations import apply_beam_to_skycomponent

log = logging.getLogger(__name__)

def find_times_above_elevation_limit(start_times, end_times, location, phasecentre, elevation_limit):
    """ Find all times for which a phasecentre is above the elevation limit
    
    :param start_times:
    :param end_times:
    :param location:
    :param phasecentre:
    :param elevation_limit:
    :return:
    """
    assert len(start_times) == len(end_times)
    
    def valid_elevation(time, location, phasecentre):
        ha = numpy.pi * time / 43200.0
        dec = phasecentre.dec.rad
        az, el = hadec_to_azel(ha, dec, location.lat.rad)
        return el > elevation_limit * numpy.pi / 180.0
    
    number_valid_times = 0
    valid_start_times = []
    for it, t in enumerate(start_times):
        if valid_elevation(start_times[it], location, phasecentre) or \
                valid_elevation(end_times[it], location, phasecentre):
            valid_start_times.append(t)
            number_valid_times += 1
    
    assert number_valid_times > 0, "No data above elevation limit"
    
    log.info("find_times_above_elevation_limit: Start times for chunks above elevation limit:")
    
    return valid_start_times


def plot_uvcoverage(vis_list, plot_file='uvcoverage.png', **kwargs):
    """ Standard plot of uv coverage
    
    :param vis_list:
    :param plot_file:
    :param kwargs:
    :return:
    """
    plt.clf()
    for ivis, vis in enumerate(vis_list):
        plt.plot(-vis.u, -vis.v, '.', color='b', markersize=0.2)
        plt.plot(vis.u, vis.v, '.', color='b', markersize=0.2)
    plt.xlabel('U (wavelengths)')
    plt.ylabel('V (wavelengths)')
    plt.title('UV coverage')
    plt.savefig(plot_file)
    plt.show(block=False)


def plot_azel(bvis_list, plot_file='azel.png', **kwargs):
    """ Standard plot of az el coverage
    
    :param bvis_list:
    :param plot_file:
    :param kwargs:
    :return:
    """
    plt.clf()
    r2d = 180.0 / numpy.pi
    for ibvis, bvis in enumerate(bvis_list):
        ha = numpy.pi * bvis.time / 43200.0
        dec = bvis.phasecentre.dec.rad
        latitude = bvis.configuration.location.lat.rad
        az, el = hadec_to_azel(ha, dec, latitude)
        if ibvis == 0:
            plt.plot(bvis.time, r2d * az, '.', color='r', label='Azimuth (deg)')
            plt.plot(bvis.time, r2d * el, '.', color='b', label='Elevation (deg)')
        else:
            plt.plot(bvis.time, r2d * az, '.', color='r')
            plt.plot(bvis.time, r2d * el, '.', color='b')
    plt.xlabel('HA (s)')
    plt.ylabel('Angle')
    plt.legend()
    plt.title('Azimuth and elevation vs hour angle')
    plt.savefig(plot_file)
    plt.show(block=False)


def plot_gaintable(gt_list, title='', plot_file='gaintable.png', **kwargs):
    """ Standard plot of gain table
    
    :param gt_list:
    :param title:
    :param plot_file:
    :param kwargs:
    :return:
    """
    plt.clf()
    for gt in gt_list:
        amp = numpy.abs(gt[0].gain[:, 0, 0, 0, 0])
        plt.plot(gt[0].time[amp > 0.0], 1.0 / amp[amp > 0.0], '.')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.savefig(plot_file)
    plt.show(block=False)

def plot_pointingtable(pt_list, plot_file, title, **kwargs):
    """ Standard plot of pointing table
    
    :param pt_list:
    :param plot_file:
    :param title:
    :param kwargs:
    :return:
    """
    plt.clf()
    r2a = 180.0 * 3600.0 / numpy.pi
    rms_az = 0.0
    rms_el = 0.0
    num = 0
    for pt in pt_list:
        num += len(pt.pointing[:, 0, 0, 0, 0])
        rms_az += numpy.sum((r2a * pt.pointing[:, 0, 0, 0, 0]) ** 2)
        rms_el += numpy.sum((r2a * pt.pointing[:, 0, 0, 0, 1]) ** 2)
        plt.plot(pt.time, r2a * pt.pointing[:, 0, 0, 0, 0], '.', color='r')
        plt.plot(pt.time, r2a * pt.pointing[:, 0, 0, 0, 1], '.', color='b')

    rms_az = numpy.sqrt(rms_az / num)
    rms_el = numpy.sqrt(rms_el / num)
    plt.title("%s az, el rms %.2f %.2f (arcsec)" % (title, rms_az, rms_el))
    plt.xlabel('Time (s)')
    plt.ylabel('Offset (arcsec)')
    plt.savefig(plot_file)
    plt.show(block=False)


def find_pb_width_null(pbtype, frequency, **kwargs):
    """ Rough estimates of HWHM and null locations
    
    :param pbtype:
    :param frequency:
    :param kwargs:
    :return:
    """
    if pbtype == 'MID':
        HWHM_deg = 0.596 * 1.36e9 / frequency[0]
        null_az_deg = 2.0 * HWHM_deg
        null_el_deg = 2.0 * HWHM_deg
    elif pbtype == 'MID_FEKO_B1':
        null_az_deg = 1.0779 * 1.36e9 / frequency[0]
        null_el_deg = 1.149 * 1.36e9 / frequency[0]
        HWHM_deg = 0.447 * 1.36e9 / frequency[0]
    elif pbtype == 'MID_FEKO_B2':
        null_az_deg = 1.0779 * 1.36e9 / frequency[0]
        null_el_deg = 1.149 * 1.36e9 / frequency[0]
        HWHM_deg = 0.447 * 1.36e9 / frequency[0]
    elif pbtype == 'MID_FEKO_Ku':
        null_az_deg = 1.0779 * 1.36e9 / frequency[0]
        null_el_deg = 1.149 * 1.36e9 / frequency[0]
        HWHM_deg = 0.447 * 1.36e9 / frequency[0]
    else:
        null_az_deg = 1.145 * 1.36e9 / frequency[0]
        null_el_deg = 1.145 * 1.36e9 / frequency[0]
        HWHM_deg = 0.447 * 1.36e9 / frequency[0]
    
    return HWHM_deg, null_az_deg, null_el_deg


def create_simulation_components(context, phasecentre, frequency, pbtype, offset_dir, flux_limit,
                                 pbradius, pb_npixel, pb_cellsize):
    """ Construct components for simulation
    
    :param context:
    :param phasecentre:
    :param frequency:
    :param pbtype:
    :param offset_dir:
    :param flux_limit:
    :param pbradius:
    :param pb_npixel:
    :param pb_cellsize:
    :return:
    """
    
    HWHM_deg, null_az_deg, null_el_deg = find_pb_width_null(pbtype, frequency)
    
    dec = phasecentre.dec.deg
    ra = phasecentre.ra.deg
    
    if context == 'singlesource':
        log.info("create_simulation_components: Constructing single component")
        offset = [HWHM_deg * offset_dir[0], HWHM_deg * offset_dir[1]]
        log.info("create_simulation_components: Offset from pointing centre = %.3f, %.3f deg" % (offset[0], offset[1]))
        
        # The point source is offset to approximately the halfpower point
        offset_direction = SkyCoord(ra=(ra + offset[0] / numpy.cos(numpy.pi * dec / 180.0)) * units.deg,
                                    dec=(dec + offset[1]) * units.deg,
                                    frame='icrs', equinox='J2000')
        
        original_components = [Skycomponent(flux=[[1.0]], direction=offset_direction, frequency=frequency,
                                            polarisation_frame=PolarisationFrame('stokesI'))]
        print(original_components[0])
    
    elif context == 'null':
        log.info("create_simulation_components: Constructing single component at the null")
        
        offset = [null_az_deg * offset_dir[0], null_el_deg * offset_dir[1]]
        HWHM = HWHM_deg * numpy.pi / 180.0
        
        log.info("create_simulation_components: Offset from pointing centre = %.3f, %.3f deg" % (offset[0], offset[1]))
        
        # The point source is offset to approximately the null point
        offset_direction = SkyCoord(ra=(ra + offset[0] / numpy.cos(numpy.pi * dec / 180.0)) * units.deg,
                                    dec=(dec + offset[1]) * units.deg,
                                    frame='icrs', equinox='J2000')
        
        original_components = [Skycomponent(flux=[[1.0]], direction=offset_direction, frequency=frequency,
                                            polarisation_frame=PolarisationFrame('stokesI'))]
        print(original_components[0])
    
    
    else:
        offset = [0.0, 0.0]
        # Make a skymodel from S3
        max_flux = 0.0
        total_flux = 0.0
        log.info("create_simulation_components: Constructing s3sky components")
        from wrappers.serial.simulation.testing_support import create_test_skycomponents_from_s3
        
        original_components = create_test_skycomponents_from_s3(flux_limit=flux_limit / 100.0,
                                                                phasecentre=phasecentre,
                                                                polarisation_frame=PolarisationFrame("stokesI"),
                                                                frequency=numpy.array(frequency),
                                                                radius=pbradius)
        log.info("create_simulation_components: %d components before application of primary beam" %
              (len(original_components)))
        
        pbmodel = create_image(npixel=pb_npixel,
                               cellsize=pb_cellsize,
                               phasecentre=phasecentre,
                               frequency=frequency,
                               polarisation_frame=PolarisationFrame(
                                   "stokesI"))
        pb = create_pb(pbmodel, "MID_GAUSS", pointingcentre=phasecentre, use_local=False)
        pb_feko = create_pb(pbmodel, pbtype, pointingcentre=phasecentre, use_local=True)
        pb.data = pb_feko.data[:, 0, ...][:, numpy.newaxis, ...]
        pb_applied_components = [copy_skycomponent(c) for c in original_components]
        pb_applied_components = apply_beam_to_skycomponent(pb_applied_components, pb)
        filtered_components = []
        for icomp, comp in enumerate(pb_applied_components):
            if comp.flux[0, 0] > flux_limit:
                total_flux += comp.flux[0, 0]
                if abs(comp.flux[0, 0]) > max_flux:
                    max_flux = abs(comp.flux[0, 0])
                filtered_components.append(original_components[icomp])
        log.info("create_simulation_components: %d components > %.3f Jy after application of primary beam" %
              (len(filtered_components), flux_limit))
        log.info("create_simulation_components: Strongest components is %g (Jy)" % max_flux)
        log.info("create_simulation_components: Total flux in components is %g (Jy)" % total_flux)
        original_components = [copy_skycomponent(c) for c in filtered_components]
        plt.clf()
        show_image(pb, components=original_components)
        plt.show(block=False)
        
        log.info("create_simulation_components: Created %d components" % len(original_components))
        # Primary beam points to the phasecentre
        offset_direction = SkyCoord(ra=ra * units.deg, dec=dec * units.deg, frame='icrs',
                                    equinox='J2000')
        
        return original_components, offset_direction



