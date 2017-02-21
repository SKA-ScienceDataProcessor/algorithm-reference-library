# Tim Cornwell <realtimcornwell@gmail.com>
#
""" Visibility iterators

"""

import logging

import numpy

from arl.data.parameters import get_parameter
from arl.data.data_models import *
from arl.data.polarisation import *

from arl.visibility.operations import create_visibility

from arl.visibility.coalesce import *

log = logging.getLogger(__name__)


def vis_timeslice_iter(vis, **kwargs):
    """ Time slice iterator
    
    If timeslice='auto' then timeslice is taken to be the difference between the first two
    unique elements of the vis time.
          
    :param timeslice: Timeslice (seconds) ('auto')
    :returns: Boolean array with selected rows=True
        
    """
    uniquetimes = numpy.unique(vis.time)
    timeslice = get_parameter(kwargs, "timeslice", 'auto')
    if timeslice == 'auto':
        log.info('vis_timeslice_iter: Found %d unique times' % len(uniquetimes))
        if len(uniquetimes) > 1:
            timeslice = (uniquetimes[1] - uniquetimes[0])
            log.debug('vis_timeslice_auto: Guessing time interval to be %.2f s' % timeslice)
        else:
            # Doesn't matter what we set it to.
            timeslice = vis.integration_time[0]
    boxes = timeslice * numpy.round(uniquetimes / timeslice).astype('int')
        
    for box in boxes:
        rows = numpy.abs(vis.time - box) < 0.5 * timeslice
        yield rows


def vis_wslice_iter(vis, wslice, **kwargs):
    """ W slice iterator

    :param wslice: wslice (wavelengths) (must be specified)
    :returns: Boolean array with selected rows=True
    """
    assert wslice is not None, "wslice must be specified"
    wmaxabs = (numpy.max(numpy.abs(vis.w)))
    nboxes = 1 + 2 * numpy.round(wmaxabs / wslice).astype('int')
    boxes = numpy.linspace(- wmaxabs, +wmaxabs, nboxes)
    
    for box in boxes:
        rows = numpy.abs(vis.w - box) < 0.5 * wslice
        if numpy.sum(rows) > 0:
            yield rows


def vis_slice_iter(vis, step, **kwargs):
    """ Iterates in slices

    :param step: Size of step to be iterated over (in rows)
    :returns: Boolean array with selected rows=True

    """
    assert step > 0
    for row in range(0, vis.nvis, step):
            yield range(row, min(row+step, vis.nvis))


def vis_create_iter(config: Configuration, times: numpy.array, freq: numpy.array, phasecentre: SkyCoord,
                    weight: float=1, npol=4, pol_frame=Polarisation_Frame('stokesI'), integration_time=1.0,
                    number_integrations=1, channel_bandwidth=1e6, coalescence_factor=1.0):
    """ Create a sequence of Visibiliites
    
    :param config: Configuration of antennas
    :param times: hour angles in radians
    :param freq: frequencies (Hz] Shape [nchan]
    :param weight: weight of a single sample
    :param phasecentre: phasecentre of observation
    :param npol: Number of polarizations
    :param integration_time: Integration time ('auto' or value in s)
    :returns: Visibility

    """
    for time in times:
        actualtimes = time + numpy.arange(0, number_integrations) * integration_time * numpy.pi / 43200.0
        vis = create_visibility(config, actualtimes, freq=freq, phasecentre=phasecentre,
                                npol=npol, pol_frame=pol_frame, weight=weight, integration_time=integration_time,
                                channel_bandwidth=channel_bandwidth)
        cvis = coalesce_visibility(vis, coalescence_factor=coalescence_factor)[0]
        yield cvis
