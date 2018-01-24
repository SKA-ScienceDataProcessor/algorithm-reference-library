"""
Functions for visibility coalescence and decoalescence.

The BlockVisibility format describes the visibility
data as it would come from the correlator: [time, ant2, ant1, channel, pol]. This is well-suited to
calibration and some visibility processing such as continuum removal. However the BlockVisibility format
is vastly oversampled on the short spacings where the visibility (after calibration) varies slowly compared to
the longest baselines. The coalescence operation resamples the visibility at a rate inversely proportional
to baseline length. This cannot be held in the BlockVIsibility format so it is stored in the Visibility
format. For e.g. SKA1-LOW, coalescing typically reduces the number of visibilities by a factor between 10 and 30.

A typical use might be::

    vt = predict_skycomponent_visibility(vt, comps)
    cvt = coalesce_visibility(vt, time_coal=1.0, max_time_coal=100, frequency_coal=0.0,
        max_frequency_coal=1)
    dirtyimage, sumwt = invert_2d(cvt, model)

"""

import numpy

from astropy import constants

from arl.visibility.base import vis_summary, copy_visibility
from arl.data.data_models import Visibility, BlockVisibility
from arl.data.parameters import get_parameter
from arl.util.array_functions import average_chunks, average_chunks2

import logging

log = logging.getLogger(__name__)


def coalesce_visibility(vis: BlockVisibility, **kwargs) -> Visibility:
    """ Coalesce the BlockVisibility data. The output format is a Visibility, as needed for imaging

    Coalesce by baseline-dependent averaging (optional). The number of integrations averaged goes as the ratio of the
    maximum possible baseline length to that for this baseline. This number can be scaled by coalescence_factor and
    limited by max_coalescence.

    When faceting, the coalescence factors should be roughly the same as the number of facets on one axis.

    If coalescence_factor=0.0 then just a format conversion is done

    :param vis: BlockVisibility to be coalesced
    :return: Coalesced visibility with  cindex and blockvis filled in
    """

    assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis

    time_coal = get_parameter(kwargs, 'time_coal', 0.0)
    max_time_coal = get_parameter(kwargs, 'max_time_coal', 100)
    frequency_coal = get_parameter(kwargs, 'frequency_coal', 0.0)
    max_frequency_coal = get_parameter(kwargs, 'max_frequency_coal', 100)

    if time_coal == 0.0 and frequency_coal == 0.0:
        return convert_blockvisibility_to_visibility((vis))

    cvis, cuvw, cwts, ctime, cfrequency, cchannel_bandwidth, ca1, ca2, cintegration_time, cindex \
        = average_in_blocks(vis.data['vis'], vis.data['uvw'], vis.data['weight'], vis.time, vis.integration_time,
                            vis.frequency, vis.channel_bandwidth, time_coal, max_time_coal,
                            frequency_coal, max_frequency_coal)
    cimwt = numpy.ones(cvis.shape)
    coalesced_vis = Visibility(uvw=cuvw, time=ctime, frequency=cfrequency,
                               channel_bandwidth=cchannel_bandwidth,
                               phasecentre=vis.phasecentre, antenna1=ca1, antenna2=ca2, vis=cvis,
                               weight=cwts, imaging_weight=cimwt,
                               configuration=vis.configuration, integration_time=cintegration_time,
                               polarisation_frame=vis.polarisation_frame, cindex=cindex,
                               blockvis=vis)

    log.debug('coalesce_visibility: Created new Visibility for coalesced data, coalescence factors (t,f) = (%.3f,%.3f)'
              % (time_coal, frequency_coal))
    log.debug('coalesce_visibility: Maximum coalescence (t,f) = (%d, %d)' % (max_time_coal, max_frequency_coal))
    log.debug('coalesce_visibility: Original %s, coalesced %s' % (vis_summary(vis),
                                                                  vis_summary(coalesced_vis)))

    return coalesced_vis


def convert_blockvisibility_to_visibility(vis: BlockVisibility) -> Visibility:
    """ Convert the BlockVisibility data with no coalescence

    :param vis: BlockVisibility to be converted
    :return: Visibility with  cindex and blockvis filled in
    """

    assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis

    cvis, cuvw, cwts, ctime, cfrequency, cchannel_bandwidth, ca1, ca2, cintegration_time, cindex \
        = convert_blocks(vis.data['vis'], vis.data['uvw'], vis.data['weight'], vis.time, vis.integration_time,
                         vis.frequency, vis.channel_bandwidth)
    cimwt = numpy.ones(cvis.shape)
    converted_vis = Visibility(uvw=cuvw, time=ctime, frequency=cfrequency,
                               channel_bandwidth=cchannel_bandwidth,
                               phasecentre=vis.phasecentre, antenna1=ca1, antenna2=ca2, vis=cvis,
                               weight=cwts, imaging_weight=cimwt,
                               configuration=vis.configuration, integration_time=cintegration_time,
                               polarisation_frame=vis.polarisation_frame, cindex=cindex,
                               blockvis=vis)

    log.debug('convert_visibility: Original %s, converted %s' % (vis_summary(vis),
                                                                 vis_summary(converted_vis)))

    return converted_vis


def decoalesce_visibility(vis: Visibility, overwrite=False, **kwargs) -> BlockVisibility:
    """ Decoalesce the visibilities to the original values (opposite of coalesce_visibility)

    This relies upon the block vis and the index being part of the vis. Needs the index generated by coalesce_visibility

    :param vis: (Coalesced visibility)
    :return: BlockVisibility with vis and weight columns overwritten
    """

    assert isinstance(vis, Visibility), "vis is not a Visibility: %r" % vis
    assert isinstance(vis.blockvis, BlockVisibility), "No blockvisibility in vis %r" % vis
    assert vis.cindex is not None, "No reverse index in Visibility %r" % vis

    if overwrite:
        log.debug('decoalesce_visibility: Created new Visibility for decoalesced data')
        decomp_vis = copy_visibility(vis.blockvis)
    else:
        log.debug('decoalesce_visibility: Filled decoalesced data into template')
        decomp_vis = vis.blockvis

    vshape = decomp_vis.data['vis'].shape

    npol = vshape[-1]
    dvis = numpy.zeros(vshape, dtype='complex')
    assert numpy.max(vis.cindex) < dvis.size
    assert numpy.max(vis.cindex) < vis.vis.shape[0], "Incorrect template used in decoalescing"
    for i in range(dvis.size // npol):
        decomp_vis.data['vis'].flat[i:i + npol] = vis.data['vis'][vis.cindex[i]]

    log.debug('decoalesce_visibility: Coalesced %s, decoalesced %s' % (vis_summary(vis),
                                                                       vis_summary(
        decomp_vis)))

    return decomp_vis


def average_in_blocks(vis, uvw, wts, times, integration_time, frequency, channel_bandwidth, time_coal=1.0,
                      max_time_coal=100, frequency_coal=1.0, max_frequency_coal=100):
    # Calculate the averaging factors for time and frequency making them the same for all times
    # for this baseline
    # Find the maximum possible baseline and then scale to this.

    # The input visibility is a block of shape [ntimes, nant, nant, nchan, npol]. We will map this
    # into rows like vis[npol] and with additional columns antenna1, antenna2, frequency

    ntimes, nant, _, nchan, npol = vis.shape

    # Pol independent weighting
    allpwtsgrid = numpy.sum(wts, axis=4)
    # Pol and frequency independent weighting
    allcpwtsgrid = numpy.sum(allpwtsgrid, axis=3)
    # Pol and time independent weighting
    alltpwtsgrid = numpy.sum(allpwtsgrid, axis=0)

    # Now calculate on a baseline basis the time and frequency averaging. We do this by looking at
    # the maximum uv distance for all data and for a given baseline. The integration time and
    # channel bandwidth are scale appropriately.
    uvmax = numpy.sqrt(numpy.max(uvw[:, 0] ** 2 + uvw[:, 1] ** 2 + uvw[:, 2] ** 2))
    time_average = numpy.ones([nant, nant], dtype='int')
    frequency_average = numpy.ones([nant, nant], dtype='int')
    ua = numpy.arange(nant)
    for a2 in ua:
        for a1 in ua:
            if allpwtsgrid[:, a2, a1, :].any() > 0.0:
                uvdist = numpy.max(numpy.sqrt(uvw[:, a2, a1, 0] ** 2 + uvw[:, a2, a1, 1] ** 2), axis=0)
                if uvdist > 0.0:
                    time_average[a2, a1] = min(max_time_coal,
                                               max(1, int(round((time_coal * uvmax / uvdist)))))
                    frequency_average[a2, a1] = min(max_frequency_coal,
                                                    max(1, int(round(frequency_coal * uvmax / uvdist))))
                else:
                    time_average[a2, a1] = max_time_coal
                    frequency_average[a2, a1] = max_frequency_coal

    # See how many time chunks and frequency we need for each baseline. To do this we use the same averaging that
    # we will use later for the actual data. This tells us the number of chunks required for each baseline.
    frequency_grid, time_grid = numpy.meshgrid(frequency, times)
    channel_bandwidth_grid, integration_time_grid = numpy.meshgrid(channel_bandwidth, integration_time)
    cnvis = 0
    time_chunk_len = numpy.ones([nant, nant], dtype='int')
    frequency_chunk_len = numpy.ones([nant, nant], dtype='int')
    for a2 in ua:
        for a1 in ua:
            if (time_average[a2, a1] > 0) & (frequency_average[a2, a1] > 0 & (allpwtsgrid[:, a2, a1, ...].any() > 0.0)):
                time_chunks, _ = average_chunks(times, allcpwtsgrid[:, a2, a1], time_average[a2, a1])
                time_chunk_len[a2, a1] = time_chunks.shape[0]
                frequency_chunks, _ = average_chunks(frequency, alltpwtsgrid[a2, a1, :], frequency_average[a2, a1])
                frequency_chunk_len[a2, a1] = frequency_chunks.shape[0]
                nrows = time_chunk_len[a2, a1] * frequency_chunk_len[a2, a1]
                cnvis += nrows

    # Now we know enough to define the output coalesced arrays. The shape will be
    # succesive a1, a2: [len_time_chunks[a2,a1], a2, a1, len_frequency_chunks[a2,a1]]
    ctime = numpy.zeros([cnvis])
    cfrequency = numpy.zeros([cnvis])
    cchannel_bandwidth = numpy.zeros([cnvis])
    cvis = numpy.zeros([cnvis, npol], dtype='complex')
    cwts = numpy.zeros([cnvis, npol])
    cuvw = numpy.zeros([cnvis, 3])
    ca1 = numpy.zeros([cnvis], dtype='int')
    ca2 = numpy.zeros([cnvis], dtype='int')
    cintegration_time = numpy.zeros([cnvis])

    # For decoalescence we keep an index to map back to the original BlockVisibility
    rowgrid = numpy.zeros([ntimes, nant, nant, nchan], dtype='int')
    rowgrid.flat = range(rowgrid.size)

    cindex = numpy.zeros([rowgrid.size], dtype='int')

    # Now go through, chunking up the various arrays. Everything is converted into an array with
    # axes [time, channel] and then it is averaged over time and frequency chunks for
    # this baseline.
    # To aid decoalescence we will need an index of which output elements a given input element
    # contributes to. This is a many to one. The decoalescence will then just consist of using
    # this index to extract the coalesced value that a given input element contributes towards.

    visstart = 0
    for a2 in ua:
        for a1 in ua:
            if (time_chunk_len[a2, a1] > 0) & (frequency_chunk_len[a2, a1] > 0) & \
                    (allpwtsgrid[:, a2, a1, :].any() > 0.0):

                nrows = time_chunk_len[a2, a1] * frequency_chunk_len[a2, a1]
                rows = slice(visstart, visstart + nrows)

                cindex.flat[rowgrid[:, a2, a1, :]] = numpy.array(range(visstart, visstart + nrows))

                ca1[rows] = a1
                ca2[rows] = a2

                # Average over time and frequency for case where polarisation isn't an issue
                def average_from_grid(arr):
                    return average_chunks2(arr, allpwtsgrid[:, a2, a1, :],
                                           (time_average[a2, a1], frequency_average[a2, a1]))[0]

                ctime[rows] = average_from_grid(time_grid).flatten()
                cfrequency[rows] = average_from_grid(frequency_grid).flatten()

                for axis in range(3):
                    uvwgrid = numpy.outer(uvw[:, a2, a1, axis], frequency / constants.c.value)
                    cuvw[rows, axis] = average_from_grid(uvwgrid).flatten()

                # For some variables, we need the sum not the average
                def sum_from_grid(arr):
                    result = average_chunks2(arr, allpwtsgrid[:, a2, a1, :],
                                             (time_average[a2, a1], frequency_average[a2, a1]))
                    return result[0] * result[0].size

                cintegration_time[rows] = sum_from_grid(integration_time_grid).flatten()
                cchannel_bandwidth[rows] = sum_from_grid(channel_bandwidth_grid).flatten()

                # For the polarisations we have to perform the time-frequency average separately for each polarisation
                for pol in range(npol):
                    result = average_chunks2(vis[:, a2, a1, :, pol], wts[:, a2, a1, :, pol],
                                             (time_average[a2, a1], frequency_average[a2, a1]))
                    cvis[rows, pol], cwts[rows, pol] = result[0].flatten(), result[1].flatten()

                visstart += nrows

    assert cnvis == visstart, "Mismatch between number of rows in coalesced visibility and index"

    return cvis, cuvw, cwts, ctime, cfrequency, cchannel_bandwidth, ca1, ca2, cintegration_time, cindex


def convert_blocks(vis, uvw, wts, times, integration_time, frequency, channel_bandwidth):
    # The input visibility is a block of shape [ntimes, nant, nant, nchan, npol]. We will map this
    # into rows like vis[npol] and with additional columns antenna1, antenna2, frequency

    ntimes, nant, _, nchan, npol = vis.shape
    assert nchan == len(frequency)

    cnvis = ntimes * nant * (nant - 1) * nchan // 2

    # Now we know enough to define the output coalesced arrays. The shape will be
    # succesive a1, a2: [len_time_chunks[a2,a1], a2, a1, len_frequency_chunks[a2,a1]]
    ctime = numpy.zeros([cnvis])
    cfrequency = numpy.zeros([cnvis])
    cchannel_bandwidth = numpy.zeros([cnvis])
    cvis = numpy.zeros([cnvis, npol], dtype='complex')
    cwts = numpy.zeros([cnvis, npol])
    cuvw = numpy.zeros([cnvis, 3])
    ca1 = numpy.zeros([cnvis], dtype='int')
    ca2 = numpy.zeros([cnvis], dtype='int')
    cintegration_time = numpy.zeros([cnvis])

    # For decoalescence we keep an index to map back to the original BlockVisibility
    rowgrid = numpy.zeros([ntimes, nant, nant, nchan], dtype='int')
    rowgrid.flat = range(rowgrid.size)

    cindex = numpy.zeros([rowgrid.size], dtype='int')

    # Now go through, chunking up the various arrays. Everything is converted into an array with
    # axes [time, channel] and then it is averaged over time and frequency chunks for
    # this baseline.
    # To aid decoalescence we will need an index of which output elements a given input element
    # contributes to. This is a many to one. The decoalescence will then just consist of using
    # this index to extract the coalesced value that a given input element contributes towards.

    row = 0
    for itime in range(ntimes):
        for a1 in range(nant):
            for a2 in range(a1 + 1, nant):
                for chan in range(nchan):
                    ca1[row] = a1
                    ca2[row] = a2
                    cfrequency[row] = frequency[chan]
                    ctime[row] = times[itime]

                    cuvw[row, :] = uvw[itime, a2, a1, :] * frequency[chan] / constants.c.value

                    cindex.flat[rowgrid[itime, a2, a1, chan]] = row
                    cintegration_time[row] = integration_time[itime]
                    cchannel_bandwidth[row] = channel_bandwidth[chan]
                    cvis[row, :] = vis[itime, a2, a1, chan, :]
                    cwts[row, :] = wts[itime, a2, a1, chan, :]
                    row += 1

    return cvis, cuvw, cwts, ctime, cfrequency, cchannel_bandwidth, ca1, ca2, cintegration_time, cindex


def decoalesce_vis(vshape, cvis, cindex):
    """Decoalesce data using Time-Baseline

    We use the index into the coalesced data. For every output row, this gives the
    corresponding row in the coalesced data.

    :param vshape: Shape of template visibility data
    :param cvis: Coalesced visibility values
    :param cindex: Index array from coalescence
    :return: uncoalesced vis
    """
    npol = vshape[-1]
    dvis = numpy.zeros(vshape, dtype='complex')
    assert numpy.max(cindex) < dvis.size
    for i in range(dvis.size // npol):
        dvis.flat[i:i + npol] = cvis[cindex[i]]

    return dvis


def convert_visibility_to_blockvisibility(vis: Visibility) -> BlockVisibility:
    """ Convert a Visibility to equivalent BlockVisibility format

    :param vis: Coalesced visibility
    :return: Visibility
    """
    if isinstance(vis, BlockVisibility):
        return vis
    else:
        return decoalesce_visibility(vis)
