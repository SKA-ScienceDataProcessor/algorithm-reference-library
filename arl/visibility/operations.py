# Tim Cornwell <realtimcornwell@gmail.com>
#
""" Visibility operations

"""

import copy
import logging

log = logging.getLogger("arl.visibility_operations")

from astropy.table import vstack

from arl.util.coordinate_support import *
from arl.data.data_models import *
from arl.data.parameters import *


def combine_visibility(vis1: Visibility, vis2: Visibility, w1: float = 1.0, w2: float = 1.0, params=None) -> Visibility:
    """ Linear combination of two visibility sets

    :param vis1: Visibility set 1
    :param vis2: Visibility set 2
    :param w1: Weight of visibility set 1
    :param w2: Weight of visibility set 2
    :param params: Dictionary containing parameters
    :returns: Visibility
    """
    if params is None:
        params = {}
    log_parameters(params)
    assert len(vis1.frequency) == len(vis2.frequency), "Visibility: frequencies should be the same"
    assert numpy.max(numpy.abs(vis1.frequency - vis2.frequency)) < 1.0, "Visibility: frequencies should be the same"
    assert len(vis1.data['vis']) == len(vis2.data['vis']), 'Length of output data table wrong'
    
    log.debug("visibility.combine: combining tables with %d rows" % (len(vis1.data)))
    log.debug("visibility.combine: weights %f, %f" % (w1, w2))
    vis = Visibility(vis=w1 * vis1.data['weight'] * vis1.data['vis'] + w2 * vis1.data['weight'] * vis2.data['vis'],
                     weight=numpy.sqrt((w1 * vis1.data['weight']) ** 2 + (w2 * vis2.data['weight']) ** 2),
                     uvw=vis1.uvw,
                     time=vis1.time,
                     antenna1=vis1.antenna1,
                     antenna2=vis1.antenna2,
                     phasecentre=vis1.phasecentre,
                     frequency=vis1.frequency,
                     configuration=vis1.configuration)
    vis.data['vis'][vis.data['weight'] > 0.0] = vis.data['vis'][vis.data['weight'] > 0.0] / \
                                                vis.data['weight'][vis.data['weight'] > 0.0]
    vis.data['vis'][vis.data['weight'] <= 0.0] = 0.0
    log.debug(u"combine_visibility: Created table with {0:d} rows".format(len(vis.data)))
    assert len(vis.data['vis']) == len(vis1.data['vis']), 'Length of output data table wrong'
    return vis


def concatenate_visibility(vis1: Visibility, vis2: Visibility, params=None) -> \
        Visibility:
    """ Concatentate the data sets in time, optionally phase rotating the second to the phasecenter of the first

    :param vis1:
    :param vis2:
    :param params: Dictionary containing parameters
    :returns: Visibility
    """
    if params is None:
        params = {}
    log_parameters(params)
    assert len(vis1.frequency) == len(vis2.frequency), "Visibility: frequencies should be the same"
    assert numpy.max(numpy.abs(vis1.frequency - vis2.frequency)) < 1.0, "Visibility: frequencies should be the same"
    log.debug(
        "visibility.concatenate: combining two tables with %d rows and %d rows" % (len(vis1.data), len(vis2.data)))
    fvis2rot = phaserotate_visibility(vis2, vis1.phasecentre)
    vis = Visibility()
    vis.data = vstack([vis1.data, fvis2rot.data], join_type='exact')
    vis.phasecentre = vis1.phasecentre
    vis.frequency = vis1.frequency
    log.debug(u"concatenate_visibility: Created table with {0:d} rows".format(len(vis.data)))
    assert (len(vis.data) == (len(vis1.data) + len(vis2.data))), 'Length of output data table wrong'
    return vis


def flag_visibility(vis: Visibility, gt: GainTable = None, params=None) -> Visibility:
    """ Flags a visibility set, optionally using GainTable

    :param vis:
    :param gt: GainTable
    :param params: Dictionary containing parameters
    :returns: Visibility
    """
    # TODO: implement
    
    if params is None:
        params = {}
    log_parameters(params)
    log.error("flag_visibility: not yet implemented")
    return vis


def filter_visibility(vis: Visibility, params=None) -> Visibility:
    """ Filter a visibility set

    :param vis:
    :param params: Dictionary containing parameters
    :returns: Visibility
    """
    # TODO: implement
    
    if params is None:
        params = {}
    log_parameters(params)
    log.error("filter_visibility: not yet implemented")
    return vis


def create_visibility(config: Configuration, times: numpy.array, freq: numpy.array, weight: float,
                      phasecentre: SkyCoord, meta: dict = None, params=None) -> Visibility:
    """ Create a Visibility from Configuration, hour angles, and direction of source

    :param params:
    :param config: Configuration of antennas
    :param times: hour angles in radians
    :param freq: frequencies (Hz] Shape [nchan, npol]
    :param weight: weight of a single sample
    :param phasecentre: phasecentre of observation
    :param meta:
    :returns: Visibility
    """
    if params is None:
        params = {}
    log_parameters(params)
    assert phasecentre is not None, "Must specify phase centre"
    nch = len(freq)
    npol = get_parameter(params, "npol", 4)
    ants_xyz = config.data['xyz']
    nants = len(config.data['names'])
    nbaselines = int(nants * (nants - 1) / 2)
    ntimes = len(times)
    nrows = nbaselines * ntimes
    row = 0
    rvis = numpy.zeros([nrows, nch, npol], dtype='complex')
    rweight = weight * numpy.ones([nrows, nch, npol])
    rtimes = numpy.zeros([nrows])
    rantenna1 = numpy.zeros([nrows], dtype='int')
    rantenna2 = numpy.zeros([nrows], dtype='int')
    for ha in times:
        rtimes[row:row + nbaselines] = ha * 43200.0 / numpy.pi
        for a1 in range(nants):
            for a2 in range(a1 + 1, nants):
                rantenna1[row] = a1
                rantenna2[row] = a2
                row += 1
    ruvw = xyz_to_baselines(ants_xyz, times, phasecentre.dec)
    log.debug(u"create_visibility: Created {0:d} rows".format(nrows))
    vis = Visibility()
    vis.data = Table(data=[ruvw, rtimes, rantenna1, rantenna2, rvis, rweight],
                     names=['uvw', 'time', 'antenna1', 'antenna2', 'vis', 'weight'], meta=meta)
    vis.frequency = freq
    vis.phasecentre = phasecentre
    vis.configuration = config
    return vis


def phaserotate_visibility(vis: Visibility, newphasecentre: SkyCoord, params=None) -> Visibility:
    """
    Phase rotate from the current phase centre to a new phase centre

    :param newphasecentre:
    :param params:
    :param vis: Visibility to be rotated
    :returns: Visibility
    """
    if params is None:
        params = {}
    log_parameters(params)
    l, m, n = skycoord_to_lmn(newphasecentre, vis.phasecentre)
    log.debug('phaserotate_visibility: Relative cartesian representation of direction = (%f, %f, '
              '%f)' % (l, m, n))
    
    # Copy object and make a new table
    vis = copy.copy(vis)
    vis.data = vis.data.copy()
    
    # No significant change?
    if numpy.abs(l) > 1e-15 or numpy.abs(m) > 1e-15:
        log.debug('phaserotate: Phase rotation from %s to %s' % (vis.phasecentre, newphasecentre))
        
        # We are going to update in-place, so make a copy
        vis.data.replace_column('vis', vis.vis.copy())
        for channel in range(vis.nchan):
            uvw = vis.uvw_lambda(channel)
            phasor = simulate_point(uvw, l, m)
            for pol in range(vis.npol):
                log.debug('phaserotate: Phaserotating visibility for channel %d, polarisation %d' %
                          (channel, pol))
                vis.vis[:, channel, pol] /= phasor
        
        # To rotate UVW, rotate into the global XYZ coordinate system and back
        xyz = uvw_to_xyz(vis.data['uvw'], ha=-vis.phasecentre.ra, dec=vis.phasecentre.dec)
        vis.data.replace_column('uvw', xyz_to_uvw(xyz, ha=-newphasecentre.ra, dec=newphasecentre.dec))
    
    vis.phasecentre = newphasecentre
    return vis


def sum_visibility(vis: Visibility, direction: SkyCoord, params=None) -> numpy.array:
    """ Direct Fourier summation in a given direction

    :param params:
    :param vis: Visibility to be summed
    :param direction: Direction of summation
    :returns: flux[nch,npol], weight[nch,pol]
    """
    if params is None:
        params = {}
    log_parameters(params)
    l, m, n = skycoord_to_lmn(direction, vis.phasecentre)
    log.debug('sum_visibility: Cartesian representation of direction = (%f, %f, %f)' % (
        l, m, n))
    flux = numpy.zeros([vis.nchan, vis.npol])
    weight = numpy.zeros([vis.nchan, vis.npol])
    for channel in range(vis.nchan):
        uvw = vis.uvw_lambda(channel)
        phasor = numpy.conj(simulate_point(uvw, l, m))
        for pol in range(vis.npol):
            log.debug('sum_visibility: Summing visibility for channel %d, polarisation %d' % (
                channel, pol))
            ws = vis.weight[:, channel, pol]
            wvis = ws * vis.vis[:, channel, pol]
            flux[channel, pol] += numpy.sum(numpy.real(wvis * phasor))
            weight[channel, pol] += numpy.sum(ws)
    flux[weight > 0.0] = flux[weight > 0.0] / weight[weight > 0.0]
    flux[weight <= 0.0] = 0.0
    return flux, weight


def aq_visibility(vis, params=None):
    """Assess the quality of Visibility

    :param params:
    :param vis: Visibility to be assessed
    :returns: AQ
    """
    if params is None:
        params = {}
    context = get_parameter(params, 'context', None)
    log_parameters(params)
    avis = numpy.abs(vis.vis)
    data = {'maxabs': numpy.max(avis),
            'minabs': numpy.min(avis),
            'rms': numpy.std(avis),
            'medianabs': numpy.median(avis)}
    qa = QA(origin=None,
            data=data,
            context=get_parameter(params, 'context', None))
    return qa
