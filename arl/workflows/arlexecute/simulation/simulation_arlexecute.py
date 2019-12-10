""" Pipelines expressed as dask components
"""

__all__ = ['simulate_list_arlexecute_workflow', 'corrupt_list_arlexecute_workflow',
           'calculate_residual_from_gaintables_arlexecute_workflow',
           'create_pointing_errors_gaintable_arlexecute_workflow',
           'create_standard_mid_simulation_arlexecute_workflow',
           'create_surface_errors_gaintable_arlexecute_workflow']

import logging

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation

from arl.data_models.memory_data_models import Visibility, SkyModel
from arl.data_models.polarisation import PolarisationFrame


from arl.processing_library.image import create_empty_image_like
from arl.processing_components.visibility import copy_visibility
from arl.processing_components.simulation.simulation_helpers import plot_pointingtable, plot_gaintable
from arl.processing_components.simulation.simulation_helpers import find_times_above_elevation_limit
from arl.processing_library.util.coordinate_support import hadec_to_azel
from arl.processing_components.calibration import apply_gaintable, create_gaintable_from_blockvisibility
from arl.wrappers.arlexecute.execution_support import arlexecute
from arl.processing_components.simulation import simulate_gaintable
from arl.processing_components.simulation import create_named_configuration
from arl.processing_components.visibility import create_blockvisibility, create_visibility
from arl.processing_components.visibility import  convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility
from arl.processing_components.calibration.pointing import create_pointingtable_from_blockvisibility
from arl.processing_components.image import import_image_from_fits
from arl.processing_components.simulation.pointing import simulate_gaintable_from_pointingtable
from arl.processing_components.simulation import simulate_pointingtable, simulate_pointingtable_from_timeseries
from arl.processing_components.simulation import create_configuration_from_MIDfile
from arl.workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow
from arl.workflows.arlexecute.skymodel.skymodel_arlexecute import predict_skymodel_list_compsonly_arlexecute_workflow

log = logging.getLogger(__name__)


def simulate_list_arlexecute_workflow(config='LOWBD2',
                                      phasecentre=SkyCoord(ra=+15.0 * u.deg, dec=-60.0 * u.deg, frame='icrs',
                                                           equinox='J2000'),
                                      frequency=None, channel_bandwidth=None, times=None,
                                      polarisation_frame=PolarisationFrame("stokesI"), order='frequency',
                                      format='blockvis',
                                      rmax=1000.0,
                                      zerow=False):
    """ A component to simulate an observation

    The simulation step can generate a single BlockVisibility or a list of BlockVisibility's.
    The parameter keyword determines the way that the list is constructed.
    If order='frequency' then len(frequency) BlockVisibility's with all times are created.
    If order='time' then  len(times) BlockVisibility's with all frequencies are created.
    If order = 'both' then len(times) * len(times) BlockVisibility's are created each with
    a single time and frequency. If order = None then all data are created in one BlockVisibility.

    The output format can be either 'blockvis' (for calibration) or 'vis' (for imaging)

    :param config: Name of configuration: def LOWBDS-CORE
    :param phasecentre: Phase centre def: SkyCoord(ra=+15.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
    :param frequency: def [1e8]
    :param channel_bandwidth: def [1e6]
    :param times: Observing times in radians: def [0.0]
    :param polarisation_frame: def PolarisationFrame("stokesI")
    :param order: 'time' or 'frequency' or 'both' or None: def 'frequency'
    :param format: 'blockvis' or 'vis': def 'blockvis'
    :return: vis_list with different frequencies in different elements
    """
    if format == 'vis':
        create_vis = create_visibility
    else:
        create_vis = create_blockvisibility
    
    if times is None:
        times = [0.0]
    if channel_bandwidth is None:
        channel_bandwidth = [1e6]
    if frequency is None:
        frequency = [1e8]
    conf = create_named_configuration(config, rmax=rmax)
    
    if order == 'time':
        log.debug("simulate_list_arlexecute_workflow: Simulating distribution in %s" % order)
        vis_list = list()
        for i, time in enumerate(times):
            vis_list.append(arlexecute.execute(create_vis, nout=1)(conf, numpy.array([times[i]]),
                                                                   frequency=frequency,
                                                                   channel_bandwidth=channel_bandwidth,
                                                                   weight=1.0, phasecentre=phasecentre,
                                                                   polarisation_frame=polarisation_frame,
                                                                   zerow=zerow))
    
    elif order == 'frequency':
        log.debug("simulate_list_arlexecute_workflow: Simulating distribution in %s" % order)
        vis_list = list()
        for j, _ in enumerate(frequency):
            vis_list.append(arlexecute.execute(create_vis, nout=1)(conf, times,
                                                                   frequency=numpy.array([frequency[j]]),
                                                                   channel_bandwidth=numpy.array(
                                                                       [channel_bandwidth[j]]),
                                                                   weight=1.0, phasecentre=phasecentre,
                                                                   polarisation_frame=polarisation_frame,
                                                                   zerow=zerow))
    
    elif order == 'both':
        log.debug("simulate_list_arlexecute_workflow: Simulating distribution in time and frequency")
        vis_list = list()
        for i, _ in enumerate(times):
            for j, _ in enumerate(frequency):
                vis_list.append(arlexecute.execute(create_vis, nout=1)(conf, numpy.array([times[i]]),
                                                                       frequency=numpy.array([frequency[j]]),
                                                                       channel_bandwidth=numpy.array(
                                                                           [channel_bandwidth[j]]),
                                                                       weight=1.0, phasecentre=phasecentre,
                                                                       polarisation_frame=polarisation_frame,
                                                                       zerow=zerow))
    
    elif order is None:
        log.debug("simulate_list_arlexecute_workflow: Simulating into single %s" % format)
        vis_list = list()
        vis_list.append(arlexecute.execute(create_vis, nout=1)(conf, times, frequency=frequency,
                                                               channel_bandwidth=channel_bandwidth,
                                                               weight=1.0, phasecentre=phasecentre,
                                                               polarisation_frame=polarisation_frame,
                                                               zerow=zerow))
    else:
        raise NotImplementedError("order $s not known" % order)
    return vis_list

def corrupt_list_arlexecute_workflow(vis_list, gt_list=None, seed=None, **kwargs):
    """ Create a graph to apply gain errors to a vis_list

    :param vis_list:
    :param gt_list: Optional gain table graph
    :param kwargs:
    :return:
    """
    
    def corrupt_vis(vis, gt, **kwargs):
        if isinstance(vis, Visibility):
            bv = convert_visibility_to_blockvisibility(vis)
        else:
            bv = vis
        if gt is None:
            gt = create_gaintable_from_blockvisibility(bv, **kwargs)
            gt = simulate_gaintable(gt, **kwargs)
            bv = apply_gaintable(bv, gt)
        
        if isinstance(vis, Visibility):
            return convert_blockvisibility_to_visibility(bv)
        else:
            return bv
    
    if gt_list is None:
        return [arlexecute.execute(corrupt_vis, nout=1)(vis_list[ivis], None, **kwargs)
                for ivis, v in enumerate(vis_list)]
    else:
        return [arlexecute.execute(corrupt_vis, nout=1)(vis_list[ivis], gt_list[ivis], **kwargs)
                for ivis, v in enumerate(vis_list)]


def calculate_residual_from_gaintables_arlexecute_workflow(sub_bvis_list, sub_components, sub_model_list,
                                                           no_error_gt_list, error_gt_list):
    """Calculate residual image corresponding to a set of gaintables

    The visibility difference for a set of components for error and no error gaintables
    are calculated and the residual images constructed

    :param sub_bvis_list:
    :param sub_components:
    :param sub_model_list:
    :param no_error_gt_list:
    :param error_gt_list:
    :return:
    """
    error_sm_list = [[
        arlexecute.execute(SkyModel, nout=1)(components=[sub_components[i]], gaintable=error_gt_list[ibv][i])
        for i, _ in enumerate(sub_components)] for ibv, bv in enumerate(sub_bvis_list)]
    
    no_error_sm_list = [[
        arlexecute.execute(SkyModel, nout=1)(components=[sub_components[i]], gaintable=no_error_gt_list[ibv][i])
        for i, _ in enumerate(sub_components)] for ibv, bv in enumerate(sub_bvis_list)]
    
    # Predict each visibility for each skymodel. We keep all the visibilities separate
    # and add up dirty images at the end of processing. We calibrate which applies the voltage pattern
    no_error_bvis_list = [arlexecute.execute(copy_visibility, nout=1)(bvis, zero=True) for bvis in sub_bvis_list]
    no_error_bvis_list = [
        predict_skymodel_list_compsonly_arlexecute_workflow(no_error_bvis_list[ibv], no_error_sm_list[ibv],
                                                            context='2d', docal=True)
        for ibv, bvis in enumerate(no_error_bvis_list)]
    
    error_bvis_list = [arlexecute.execute(copy_visibility, nout=1)(bvis, zero=True) for bvis in sub_bvis_list]
    error_bvis_list = [predict_skymodel_list_compsonly_arlexecute_workflow(error_bvis_list[ibv], error_sm_list[ibv],
                                                                           context='2d', docal=True)
                       for ibv, bvis in enumerate(error_bvis_list)]
    
    # Inner nest is bvis per skymodels, outer is over vis's. Calculate residual visibility
    def subtract_vis_convert(error_bvis, no_error_bvis):
        error_bvis.data['vis'] = error_bvis.data['vis'] - no_error_bvis.data['vis']
        error_vis = convert_blockvisibility_to_visibility(error_bvis)
        return error_vis
    
    error_vis_list = [[arlexecute.execute(subtract_vis_convert)(error_bvis_list[ibvis][icomp],
                                                                no_error_bvis_list[ibvis][icomp])
                       for icomp, _ in enumerate(sub_components)]
                      for ibvis, _ in enumerate(error_bvis_list)]
    
    # Now for each visibility/component, we make the component dirty images. We just add these
    # component dirty images since the weights should be the same
    def sum_images(images):
        sum_image = create_empty_image_like(images[0][0])
        for im in images:
            sum_image.data += im[0].data
        return sum_image, images[0][1]
    
    dirty_list = list()
    for vis in error_vis_list:
        result = invert_list_arlexecute_workflow(vis, sub_model_list, '2d')
        dirty_list.append(arlexecute.execute(sum_images)(result))
    
    return dirty_list

def create_pointing_errors_gaintable_arlexecute_workflow(sub_bvis_list, sub_components, sub_vp_list,
                                                         use_radec=False, pointing_error=0.0, static_pointing_error=None,
                                                         global_pointing_error=None, time_series='', time_series_type='',
                                                         seed=None, pointing_directory=None, show=False, basename=''):
    if global_pointing_error is None:
        global_pointing_error = [0.0, 0.0]
    
    # One pointing table per visibility
    
    error_pt_list = [arlexecute.execute(create_pointingtable_from_blockvisibility)(bvis) for bvis in sub_bvis_list]
    no_error_pt_list = [arlexecute.execute(create_pointingtable_from_blockvisibility)(bvis) for bvis in sub_bvis_list]
    
    if time_series is '':
        error_pt_list = [arlexecute.execute(simulate_pointingtable)(pt, pointing_error=pointing_error,
                                                                    static_pointing_error=static_pointing_error,
                                                                    global_pointing_error=global_pointing_error,
                                                                    seed=seed)
                         for ipt, pt in enumerate(error_pt_list)]
    else:
        error_pt_list = [arlexecute.execute(simulate_pointingtable_from_timeseries)(pt, type=time_series,
                                                                                    time_series_type=time_series_type,
                                                                                    pointing_directory=pointing_directory,
                                                                                    seed=seed)
                         for ipt, pt in enumerate(error_pt_list)]
    
    if show:
        tmp_error_pt_list = arlexecute.compute(error_pt_list, sync=True)
        if time_series != "":
            plot_file = 'pointing_error_%s.png' % (time_series_type)
        else:
            r2s = 180 * 3600.0 / numpy.pi
            plot_file = 'pointing_error_dynamic_%.2f_static_(%.2f,%.2f)_global_(%.2f,%.2f).png' % \
                        (r2s * pointing_error, r2s * static_pointing_error[0], r2s * static_pointing_error[1],
                         r2s * global_pointing_error[0], r2s * global_pointing_error[1])
        
        plot_pointingtable(tmp_error_pt_list, plot_file=plot_file, title=basename)
    
    # Create the gain tables, one per Visibility and per component
    no_error_gt_list = [arlexecute.execute(simulate_gaintable_from_pointingtable)
                        (bvis, sub_components, no_error_pt_list[ibv], sub_vp_list[ibv], use_radec=use_radec)
                        for ibv, bvis in enumerate(sub_bvis_list)]
    error_gt_list = [arlexecute.execute(simulate_gaintable_from_pointingtable)
                     (bvis, sub_components, error_pt_list[ibv], sub_vp_list[ibv], use_radec=use_radec)
                     for ibv, bvis in enumerate(sub_bvis_list)]
    if show:
        tmp_gt_list = arlexecute.compute(error_gt_list, sync=True)
        
        if time_series_type != "":
            plot_file = 'gaintable_%s.png' % time_series_type
        else:
            r2s = 180 * 3600.0 / numpy.pi
            plot_file = 'gaintable_dynamic_%.2f_static_(%.2f,%.2f)_global_(%.2f,%.2f).png' % \
                        (r2s * pointing_error, r2s * static_pointing_error[0], r2s * static_pointing_error[1],
                         r2s * global_pointing_error[0], r2s * global_pointing_error[1])
        
        plot_gaintable(tmp_gt_list, title="%s: dish 0 amplitude gain, %s" % (basename, time_series_type),
                       plot_file=plot_file)
    
    return no_error_gt_list, error_gt_list

def create_surface_errors_gaintable_arlexecute_workflow(band, sub_bvis_list, sub_components, vp_directory, use_radec=False,
                                                        elevation_sampling=5.0, show=False, basename=''):
    def get_band_vp(band, el):
        
        if band == 'B1':
            vpa = import_image_from_fits('%s/B1_%d_0565_real_interpolated.fits' % (vp_directory, int(el)))
            vpa_imag = import_image_from_fits('%s/B1_%d_0565_imag_interpolated.fits' % (vp_directory, int(el)))
        elif band == 'B2':
            vpa = import_image_from_fits('%s/B2_%d_1360_real_interpolated.fits' % (vp_directory, int(el)))
            vpa_imag = import_image_from_fits('%s/B2_%d_1360_imag_interpolated.fits' % (vp_directory, int(el)))
        elif band == 'Ku':
            vpa = import_image_from_fits('%s/Ku_%d_11700_real_interpolated.fits' % (vp_directory, int(el)))
            vpa_imag = import_image_from_fits('%s/Ku_%d_11700_imag_interpolated.fits' % (vp_directory, int(el)))
        else:
            raise ValueError("Unknown band %s" % band)
        
        vpa.data = vpa.data + 1j * vpa_imag.data
        return vpa
    
    def find_vp(band, vis):
        ha = numpy.pi * numpy.average(vis.time) / 43200.0
        dec = vis.phasecentre.dec.rad
        latitude = vis.configuration.location.lat.rad
        az, el = hadec_to_azel(ha, dec, latitude)
        
        el_deg = el * 180.0 / numpy.pi
        el_table = max(0.0,
                       min(90.1, elevation_sampling * ((el_deg + elevation_sampling / 2.0) // elevation_sampling)))
        return get_band_vp(band, el_table)
    
    def find_vp_nominal(band):
        el_nominal_deg = 45.0
        return get_band_vp(band, el_nominal_deg)
    
    error_pt_list = [arlexecute.execute(create_pointingtable_from_blockvisibility)(bvis) for bvis in sub_bvis_list]
    no_error_pt_list = [arlexecute.execute(create_pointingtable_from_blockvisibility)(bvis) for bvis in sub_bvis_list]
    
    vp_nominal_list = [arlexecute.execute(find_vp_nominal)(band) for bv in sub_bvis_list]
    vp_actual_list = [arlexecute.execute(find_vp)(band, bv) for bv in sub_bvis_list]
    
    # Create the gain tables, one per Visibility and per component
    no_error_gt_list = [arlexecute.execute(simulate_gaintable_from_pointingtable)
                        (bvis, sub_components, no_error_pt_list[ibv], vp_nominal_list[ibv], use_radec=use_radec)
                        for ibv, bvis in enumerate(sub_bvis_list)]
    error_gt_list = [arlexecute.execute(simulate_gaintable_from_pointingtable)
                     (bvis, sub_components, error_pt_list[ibv], vp_actual_list[ibv], use_radec=use_radec)
                     for ibv, bvis in enumerate(sub_bvis_list)]
    if show:
        plot_file = 'gaintable.png'
        tmp_gt_list = arlexecute.compute(error_gt_list, sync=True)
        plot_gaintable(tmp_gt_list, plot_file=plot_file, title=basename)
    
    return no_error_gt_list, error_gt_list

def create_standard_mid_simulation_arlexecute_workflow(band, rmax, phasecentre, time_range, time_chunk, integration_time,
                                                       shared_directory):
    """ Create the standard MID simulation
    
    :param band:
    :param rmax:
    :param ra:
    :param declination:
    :param time_range:
    :param time_chunk:
    :param integration_time:
    :param shared_directory:
    :return:
    """
    
    # Set up details of simulated observation
    if band == 'B1':
        frequency = [0.765e9]
    elif band == 'B2':
        frequency = [1.36e9]
    elif band == 'Ku':
        frequency = [12.179e9]
    else:
        raise ValueError("Unknown band %s" % band)
    
    channel_bandwidth = [1e7]
    mid_location = EarthLocation(lon="21.443803", lat="-30.712925", height=0.0)
    
    # Do each time_chunk in parallel
    start_times = numpy.arange(time_range[0] * 3600, time_range[1] * 3600, time_chunk)
    end_times = start_times + time_chunk
    
    start_times = find_times_above_elevation_limit(start_times, end_times, location=mid_location,
                                                   phasecentre=phasecentre, elevation_limit=15.0)
    times = [numpy.arange(start_times[itime], end_times[itime], integration_time) for itime in
             range(len(start_times))]
    
    s2r = numpy.pi / (12.0 * 3600)
    rtimes = s2r * numpy.array(times)
    ntimes = len(rtimes.flat)
    nchunks = len(start_times)
    
    assert ntimes > 0, "No data above elevation limit"
    
    #print('%d integrations of duration %.1f s processed in %d chunks' % (ntimes, integration_time, nchunks))
    
    mid = create_configuration_from_MIDfile('%s/ska1mid_local.cfg' % shared_directory, rmax=rmax,
                                            location=mid_location)
    
    bvis_graph = [arlexecute.execute(create_blockvisibility)(mid, rtimes[itime], frequency=frequency,
                                                             channel_bandwidth=channel_bandwidth, weight=1.0,
                                                             phasecentre=phasecentre,
                                                             polarisation_frame=PolarisationFrame("stokesI"),
                                                             zerow=True)
                  for itime in range(nchunks)]
    
    return bvis_graph
