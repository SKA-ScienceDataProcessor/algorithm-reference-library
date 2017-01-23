# Tim Cornwell <realtimcornwell@gmail.com>
#
""" Visibility operations

"""

import sys
import copy
import logging

from arl.util.coordinate_support import *
from arl.data.data_models import *
from arl.data.parameters import *

log = logging.getLogger(__name__)

def vis_summary(vis: Visibility):
    """Return string summarizing the Visibility
    
    """
    return "Visibility has %d rows, total size %.3f GB" % (vis.nvis, vis.__sizeof__())

def combine_visibility(vis1: Visibility, vis2: Visibility, w1: float = 1.0, w2: float = 1.0, **kwargs) -> Visibility:
    """ Linear combination of two visibility sets

    :param vis1: Visibility set 1
    :param vis2: Visibility set 2
    :param w1: Weight of visibility set 1
    :param w2: Weight of visibility set 2
    :param params: Dictionary containing parameters
    :returns: Visibility
    """
    assert len(vis1.frequency) == len(vis2.frequency), "Visibility: frequencies should be the same"
    assert numpy.max(numpy.abs(vis1.frequency - vis2.frequency)) < 1.0, "Visibility: frequencies should be the same"
    assert len(vis1.data['vis']) == len(vis2.data['vis']), 'Length of output data table wrong'
    
    log.info("combine_visibility: combining tables with %d rows" % (len(vis1.data)))
    log.info("combine_visibility: weights %f, %f" % (w1, w2))
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
    log.info("combine_visibility: %s" % (vis_summary(vis)))
    assert len(vis.data['vis']) == len(vis1.data['vis']), 'Length of output data table wrong'
    return vis


def concatenate_visibility(vis1: Visibility, vis2: Visibility) -> \
        Visibility:
    """ Concatentate the data sets in time, optionally phase rotating the second to the phasecenter of the first

    :param vis1:
    :param vis2:
    :param params: Dictionary containing parameters
    :returns: Visibility
    """
    assert len(vis1.frequency) == len(vis2.frequency), "Visibility: frequencies should be the same"
    assert numpy.max(numpy.abs(vis1.frequency - vis2.frequency)) < 1.0, "Visibility: frequencies should be the same"
    log.info(
        "concatenate_visibility: combining two tables with %d rows and %d rows" % (len(vis1.data), len(vis2.data)))
    fvis2rot = phaserotate_visibility(vis2, vis1.phasecentre)
    vis = Visibility()
    vis.data = vstack([vis1.data, fvis2rot.data], join_type='exact')
    vis.phasecentre = vis1.phasecentre
    vis.frequency = vis1.frequency
    log.info("concatenate_visibility: %s" % (vis_summary(vis)))
    assert (len(vis.data) == (len(vis1.data) + len(vis2.data))), 'Length of output data table wrong'
    return vis


def create_visibility(config: Configuration, times: numpy.array, freq: numpy.array, phasecentre: SkyCoord,
                      weight: float, meta: dict = None, npol=4) -> Visibility:
    """ Create a Visibility from Configuration, hour angles, and direction of source

    :param params:
    :param config: Configuration of antennas
    :param times: hour angles in radians
    :param freq: frequencies (Hz] Shape [nchan]
    :param weight: weight of a single sample
    :param phasecentre: phasecentre of observation
    :param npol: Number of polarizations
    :returns: Visibility
    """
    assert phasecentre is not None, "Must specify phase centre"
    nch = len(freq)
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
    ruvw = xyz_to_baselines(ants_xyz, times, phasecentre.dec.value)
    vis = Visibility(uvw=ruvw, time=rtimes, antenna1=rantenna1, antenna2=rantenna2, vis=rvis, weight=rweight,
                          imaging_weight=rweight)
    vis.frequency = freq
    vis.phasecentre = phasecentre
    vis.configuration = config
    log.info("create_visibility: %s" % (vis_summary(vis)))
    return vis


def create_visibility_from_rows(vis: Visibility, rows) -> Visibility:
    """ Create a Visibility from selected rows

    :param params:
    :param vis: Visibility
    :param rows: Boolean array of row selction
    :returns: Visibility
    """

    newvis = Visibility(data=vis.data[rows], phasecentre=vis.phasecentre, configuration=vis.configuration,
                        frequency=vis.frequency)
    
    log.info("create_visibility_from_rows: Created view into visibility table")
    return newvis


def phaserotate_visibility(vis: Visibility, newphasecentre: SkyCoord, tangent=True, inverse=False,
                           **kwargs) -> Visibility:
    """
    Phase rotate from the current phase centre to a new phase centre

    :param tangent:
    :param newphasecentre:
    :param params:
    :param inverse: Actually do the opposite
    :param vis: Visibility to be rotated
    :returns: Visibility
    """
    l, m, n = skycoord_to_lmn(newphasecentre, vis.phasecentre)
    
    # No significant change?
    if numpy.abs(l) > 1e-15 or numpy.abs(m) > 1e-15:
    
        # Copy object and make a new table
        vis = copy.copy(vis)
        vis.data = vis.data.copy()
    
        # We are going to update in-place, so make a copy
        # vis.data.replace_column('vis', vis.vis.copy())
        for channel in range(vis.nchan):
            uvw = vis.uvw_lambda(channel)
            phasor = simulate_point(uvw, l, m)
            if inverse:
                for pol in range(vis.npol):
                    vis.vis[:, channel, pol] *= phasor
            else:
                for pol in range(vis.npol):
                    vis.vis[:, channel, pol] *= numpy.conj(phasor)

        # To rotate UVW, rotate into the global XYZ coordinate system and back. We have the option of
        # staying on the tangent plane or not. If we stay on the tangent then the raster will
        # join smoothly at the edges. If we change the tangent then we will have to reproject to get
        # the results on the same image, in which case overlaps or gaps are difficult to deal with.
        if not tangent:
            if inverse:
                xyz = uvw_to_xyz(vis.data['uvw'], ha=-vis.phasecentre.ra, dec=vis.phasecentre.dec)
                vis.data['uvw'][...] = xyz_to_uvw(xyz, ha=-newphasecentre.ra, dec=newphasecentre.dec)[...]
            else:
                # This is the original (non-inverse) code
                xyz = uvw_to_xyz(vis.data['uvw'], ha=-vis.phasecentre.ra, dec=vis.phasecentre.dec)
                vis.data['uvw'][...] = xyz_to_uvw(xyz, ha=-newphasecentre.ra, dec=newphasecentre.dec)[...]
    else:
        log.warning("phaserotate_visibility: Null phase rotation")

    vis.phasecentre = newphasecentre

    return vis


def sum_visibility(vis: Visibility, direction: SkyCoord) -> numpy.array:
    """ Direct Fourier summation in a given direction

    :param params:
    :param vis: Visibility to be summed
    :param direction: Direction of summation
    :returns: flux[nch,npol], weight[nch,pol]
    """

    l, m, n = skycoord_to_lmn(direction, vis.phasecentre)
    flux = numpy.zeros([vis.nchan, vis.npol])
    weight = numpy.zeros([vis.nchan, vis.npol])
    for channel in range(vis.nchan):
        uvw = vis.uvw_lambda(channel)
        phasor = numpy.conj(simulate_point(uvw, l, m))
        for pol in range(vis.npol):
            ws = vis.weight[:, channel, pol]
            wvis = ws * vis.vis[:, channel, pol]
            flux[channel, pol] += numpy.sum(numpy.real(wvis * phasor))
            weight[channel, pol] += numpy.sum(ws)
    flux[weight > 0.0] = flux[weight > 0.0] / weight[weight > 0.0]
    flux[weight <= 0.0] = 0.0
    return flux, weight


def qa_visibility(vis, context = None):
    """Assess the quality of Visibility

    :param params:
    :param vis: Visibility to be assessed
    :returns: AQ
    """
    avis = numpy.abs(vis.vis)
    data = {'maxabs': numpy.max(avis),
            'minabs': numpy.min(avis),
            'rms': numpy.std(avis),
            'medianabs': numpy.median(avis)}
    qa = QA(origin=None,
            data=data,
            context=context)
    return qa
