""" Pipelines expressed as dask components
"""

__all__ = ['simulate_list_serial_workflow', 'corrupt_list_serial_workflow']

import logging

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from data_models.memory_data_models import  Visibility, BlockVisibility


from wrappers.serial.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility
from wrappers.serial.simulation.testing_support import simulate_gaintable
from processing_components.simulation.configurations import create_named_configuration
from wrappers.serial.visibility.base import create_blockvisibility, create_visibility
from wrappers.serial.visibility.coalesce import convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility

log = logging.getLogger(__name__)


def simulate_list_serial_workflow(config='LOWBD2',
                                      phasecentre=SkyCoord(ra=+15.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000'),
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
        log.debug("simulate_list_serial_workflow: Simulating distribution in %s" % order)
        vis_list = list()
        for i, time in enumerate(times):
            vis_list.append(create_vis(conf, numpy.array([times[i]]),
                                                                   frequency=frequency,
                                                                   channel_bandwidth=channel_bandwidth,
                                                                   weight=1.0, phasecentre=phasecentre,
                                                                   polarisation_frame=polarisation_frame,
                                                                   zerow=zerow))
    
    elif order == 'frequency':
        log.debug("simulate_list_serial_workflow: Simulating distribution in %s" % order)
        vis_list = list()
        for j, _ in enumerate(frequency):
            vis_list.append(create_vis(conf, times,
                                                                   frequency=numpy.array([frequency[j]]),
                                                                   channel_bandwidth=numpy.array(
                                                                       [channel_bandwidth[j]]),
                                                                   weight=1.0, phasecentre=phasecentre,
                                                                   polarisation_frame=polarisation_frame,
                                                                   zerow=zerow))
    
    elif order == 'both':
        log.debug("simulate_list_serial_workflow: Simulating distribution in time and frequency")
        vis_list = list()
        for i, _ in enumerate(times):
            for j, _ in enumerate(frequency):
                vis_list.append(create_vis(conf, numpy.array([times[i]]),
                                                                       frequency=numpy.array([frequency[j]]),
                                                                       channel_bandwidth=numpy.array(
                                                                           [channel_bandwidth[j]]),
                                                                       weight=1.0, phasecentre=phasecentre,
                                                                       polarisation_frame=polarisation_frame,
                                                                       zerow=zerow))
    
    elif order is None:
        log.debug("simulate_list_serial_workflow: Simulating into single %s" % format)
        vis_list = list()
        vis_list.append(create_vis(conf, times, frequency=frequency,
                                                               channel_bandwidth=channel_bandwidth,
                                                               weight=1.0, phasecentre=phasecentre,
                                                               polarisation_frame=polarisation_frame,
                                                               zerow=zerow))
    else:
        raise NotImplementedError("order $s not known" % order)
    return vis_list


def corrupt_list_serial_workflow(vis_list, gt_list=None, seed=None, **kwargs):
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
        return [corrupt_vis(vis_list[ivis], None, **kwargs)
                for ivis, v in enumerate(vis_list)]
    else:
        return [corrupt_vis(vis_list[ivis], gt_list[ivis], **kwargs)
                for ivis, v in enumerate(vis_list)]
    
