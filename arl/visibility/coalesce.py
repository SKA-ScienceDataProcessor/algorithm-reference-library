# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing. These are built on top of the core
functions in arl.fourier_transforms.
"""

import copy

from arl.data.data_models import *
from arl.data.parameters import get_parameter
from arl.util.array_functions import average_chunks2
from arl.visibility.operations import vis_summary, copy_visibility

log = logging.getLogger(__name__)


def coalesce_visibility(vis, **kwargs):
    """ Coalesce the visibility data

    'tb': Coalesce by baseline-dependent averaging. The number of integrations averaged goes as the ration of the
    maximum possible baseline length to that for this baseline. This number can be scaled by coalescence_factor and
    limited by max_coalescence.

    'uv': Coalesce by gridding the visibilities onto a fine grid and then extracting the visibilities, weights and uvw
    from the grid. The maximum number of rows in the output visibility is the same number as the number of pixels
    in each polarisation-frequency plane i.e. nx, ny

    :param vis: Visibility to be coalesced
    :returns: Coalesced visibility, dindex
    """
    # No-op if already coalesced

    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis

    cindex = None
    # Default is no-op
    
    coalescence_factor = get_parameter(kwargs, "coalescence_factor", 0.0)
    if coalescence_factor > 0.0:
        max_coalescence    = get_parameter(kwargs, "max_coalescence", 10)
        cvis, cuvw, cvisweights, ctime, cfrequency, cpolarisation, ca1, ca2, cintegration_time, cindex = \
            coalesce_tbgrid_vis(vis.data['vis'], vis.data['time'], vis.data['frequency'],
                                vis.data['polarisation'],
                                vis.data['antenna1'], vis.data['antenna2'],
                                vis.data['uvw'], vis.data['weight'], vis.data['integration_time'],
                                max_coalescence=max_coalescence, coalescence_factor=coalescence_factor)
        
        cimwt = numpy.ones(cvis.shape)
        nrows = cvis.shape[0]
        cintegration_time = numpy.ones(nrows)
        cchannel_width = cfrequency
        coalesced_vis = Visibility(uvw=cuvw, time=ctime, frequency=cfrequency,
                                              channel_bandwidth=cchannel_width, polarisation=cpolarisation,
                                              phasecentre=vis.phasecentre, antenna1=ca1, antenna2=ca2, vis=cvis,
                                              weight=cvisweights, imaging_weight=cimwt,
                                              configuration=vis.configuration, integration_time=cintegration_time)
        
        log.info('coalesce_visibility: Created new Visibility for coalesced data, coalescence factor = %.3f' % (
            coalescence_factor))
        log.info('coalesce_visibility: Original %s, coalesced %s' % (vis_summary(vis), vis_summary(coalesced_vis)))
    else:
        return vis, None

    return coalesced_vis, cindex


def decoalesce_visibility(vis, template_vis, cindex=None, **kwargs):
    """ Decoalesce the visibilities to the original values (opposite of coalesce_visibility)
    
    The template Visibility must always be given. This is the Visibility that was coalesced.
    
    'uv': Needs the original image used in coalesce_visibility
    'tb': Needs the index generated by coalesce_visibility

    :param vis: (Coalesced visibility)
    :param template_vis: Template visibility to be filled in
    :param cindex: Index created by coalesce
    :returns: New visibility with vis and weight columns overwritten
    """

    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis

    assert type(template_vis) is Visibility, "template_vis is not a Visibility: %r" % vis

    if cindex is None:
        return vis
    
    log.info('decoalesce_visibility: Created new Visibility for decoalesced data')
    
    decomp_vis = copy_visibility(template_vis)

    decomp_vis.data['vis'] = \
        decoalesce_tbgrid_vis(template_vis.data['vis'].shape, vis.data['vis'], cindex)

    log.info('decoalesce_visibility: Coalesced %s, decoalesced %s' % (vis_summary(vis), vis_summary(decomp_vis)))
    
    return decomp_vis


def coalesce_tbgrid_vis(vis, time, frequency, polarisation, antenna1, antenna2, uvw, visweights, integration_time,
                        max_coalescence=10,
                        coalescence_factor=1.0):
    """Coalesce data by gridding onto a time baseline grid

    :param shape: Shape of grid to be used => shape of output visibilities
    :param uv: Input UVW positions
    :param vis: Visibility values
    :param visweights: Visibility weights
    :param max_coalescence: Maximum number of elements to average
    :param coalescence_factor: Boost factor for coalescence > 1 implies more averaging
    :returns: vis, uvw, visweights
    """

    # Find the maximum possible baseline and then scale to this.
    uvmax = numpy.sqrt(numpy.max(uvw[:, 0] ** 2 + uvw[:, 1] ** 2 + uvw[:, 1] ** 2))
    
    
    nvis = vis.shape[0]
    
    nant = numpy.max(antenna2) + 1
    nbaselines = nant * (nant - 1)
    
    # We first contruct maps to unique inputs e.g. for times i.e. 0 refers to the first unique time, 1 to the second
    # unique time
    timemap, utime = construct_map(time)
    ntime = len(utime)

    integration_timemap, uintegration_time = construct_map(integration_time)

    frequencymap, ufrequency = construct_map(frequency)
    nfrequency = len(ufrequency)
    
    polarisationmap, upolarisation = construct_map(polarisation)
    npolarisation = len(upolarisation)

    log.info('coalesce_tbgrid_vis: Coalescing %d unique times, %d frequencies and %d baselines' % (ntime,
                                                                                                    nfrequency,
                                                                                                    nbaselines))
    # Now that we have the maps, we can define grids to hold the various data
    timeshape = nant, nant, ntime, nfrequency
    timegrid = numpy.zeros(timeshape)

    time_integration = utime[1] - utime[0]
    log.info('coalesce_tbgrid_vis: Time step between integrations seems to be %.2f (seconds)' % time_integration)

    integration_time_grid = numpy.zeros(timeshape)
    
    frequencyshape = nant, nant, ntime, nfrequency
    frequencygrid = numpy.zeros(frequencyshape)

    visshape = nant, nant, ntime, nfrequency, npolarisation
    visgrid = numpy.zeros(visshape, dtype='complex')
    wtsgrid = numpy.zeros(visshape)

    uvwshape = nant, nant, ntime, nfrequency, 3
    uvwgrid = numpy.zeros(uvwshape)

    coords = timemap, frequencymap, polarisationmap, antenna2, antenna1
  
    # Assume that a2 > a1. We rearrange the visibility data to be per time per baseline/frequency/polarisation
    # i.e. antenna2, antenna1, time, chan, pol instead of
    #      time, antenna2, antenna2, chan, pol
    for v, wt, tm, fm, pm, a2, a1 in zip(vis, visweights, *coords):
        visgrid[a2, a1, tm, fm, pm] = v
        wtsgrid[a2, a1, tm, fm, pm] = wt

    # For some uses, we need to know that there are any visibilities
    allpwtsgrid = numpy.sum(wtsgrid, axis=4)
    
    # To facilitate decoalescence, we will keep the mapping between input row numbers
    rowgrid = numpy.zeros(timeshape, dtype='int')
    
    rows = range(nvis)
    coords = rows, timemap, time, frequencymap, frequency, integration_timemap, integration_time, \
             antenna2, antenna1, uvw[:, 0], uvw[:, 1], uvw[:, 2]
    for row, tm, tg, fm, fg, itm, it, a2, a1, uu, vv, ww in zip(*coords):
        uvwgrid[a2, a1, tm, fm, 0] = uu
        uvwgrid[a2, a1, tm, fm, 1] = vv
        uvwgrid[a2, a1, tm, fm, 2] = ww
        timegrid[a2, a1, tm, fm] = tg
        frequencygrid[a2, a1, tm, fm] = fg
        integration_time_grid[a2, a1, tm, fm] = itm
        rowgrid[a2, a1, tm, fm] = row

    visgrid[wtsgrid > 0.0] /= wtsgrid[wtsgrid > 0.0]
    visgrid[wtsgrid <= 0.0] = 0.0
    
    # Calculate the averaging factors for time and frequency making them the same for all times
    # for this baseline
    time_average = numpy.zeros([nant, nant], dtype='int')
    frequency_average = numpy.zeros([nant, nant], dtype='int')
    ua2 = numpy.unique(antenna2)
    ua1 = numpy.unique(antenna1)
    for a2 in ua2:
        for a1 in ua1:
            if (a1 < a2) & (allpwtsgrid[a2, a1, ...].any() > 0.0):
                uvdist = numpy.max(numpy.sqrt(uvwgrid[a2, a1, :, :, 0] ** 2 + uvwgrid[a2, a1, :, :, 1] ** 2))
                if uvdist > 0.0:
                    time_average[a2, a1]      = min(max_coalescence,
                                                    max(1, int(round((coalescence_factor * uvmax / uvdist)))))
                    frequency_average[a2, a1] = min(max_coalescence,
                                                    max(1, int(round((coalescence_factor * uvmax / uvdist)))))

    # See how many time chunks and frequency we need for each baseline. To do this we use the same averaging that
    # we will use later for the actual data. This tells us the number of chunks required for each baseline.
    cnvis = 0
    len_time_chunks = numpy.ones([nant, nant], dtype='int')
    len_frequency_chunks = numpy.ones([nant, nant], dtype='int')
    for a2 in ua2:
        for a1 in ua1:
            if (a1 < a2) & (time_average[a2,a1] > 0) & (frequency_average[a2,a1] > 0):
                time_chunks, wts = average_chunks2(timegrid[a2, a1, :, :],
                                                   allpwtsgrid[a2, a1, :, :],
                                                   (time_average[a2, a1], frequency_average[a2, a1]))
                len_time_chunks[a2, a1] = time_chunks.shape[0]
                len_frequency_chunks[a2, a1] = time_chunks.shape[1]

                cnvis += len_time_chunks[a2, a1] * len_frequency_chunks[a2, a1]
   
    # Now we know enough to define the output coalesced arrays
    ctime = numpy.zeros([cnvis])
    cfrequency = numpy.zeros([cnvis])
    cpolarisation = numpy.zeros([cnvis], dtype='int')
    cvis = numpy.zeros([cnvis], dtype='complex')
    cwts = numpy.zeros([cnvis])
    cuvw = numpy.zeros([cnvis, 3])
    ca1 = numpy.zeros([cnvis], dtype='int')
    ca2 = numpy.zeros([cnvis], dtype='int')
    cintegration_time = numpy.zeros([cnvis])

    # Now go through, chunking up the various arrays. As written this produces sort order with a2, a1 varying slowest
    cindex = numpy.zeros([nvis], dtype='int')
    visstart = 0
    for a2 in ua2:
        for a1 in ua1:
            if (a1 < a2) & (len_time_chunks[a2,a1] > 0) & (len_frequency_chunks[a2,a1] > 0):
                nrows = len_time_chunks[a2, a1] * len_frequency_chunks[a2, a1]
                rows = slice(visstart, visstart + nrows)
                cindex[rowgrid[a2,a1,:]] = numpy.array(range(visstart, visstart + nrows))

                ca1[rows] = a1
                ca2[rows] = a2
                cpolarisation[rows] = 0
                
                cintegration_time[rows] = average_chunks2(integration_time_grid[a2, a1, :, :],
                                                          allpwtsgrid[a2, a1, :, :],
                                                          (time_average[a2, a1], frequency_average[a2, a1]))[0].flatten()

                ctime[rows] = average_chunks2(timegrid[a2, a1, :, :], allpwtsgrid[a2, a1, :, :],
                                                 (time_average[a2, a1], frequency_average[a2, a1]))[0].flatten()
                ctime[rows] = average_chunks2(timegrid[a2, a1, :, :], allpwtsgrid[a2, a1, :, :],
                                                 (time_average[a2, a1], frequency_average[a2, a1]))[0].flatten()
                cfrequency[rows] = average_chunks2(frequencygrid[a2, a1, :, :], allpwtsgrid[a2, a1, :, :],
                                                 (time_average[a2, a1], frequency_average[a2, a1]))[0].flatten()

                cuvw[rows, 0] = average_chunks2(uvwgrid[a2, a1, :, :, 0], allpwtsgrid[a2, a1, :, :],
                                                   (time_average[a2, a1], frequency_average[a2, a1]))[0].flatten()
                cuvw[rows, 1] = average_chunks2(uvwgrid[a2, a1, :, :, 1], allpwtsgrid[a2, a1, :, :],
                                                   (time_average[a2, a1], frequency_average[a2, a1]))[0].flatten()
                cuvw[rows, 2] = average_chunks2(uvwgrid[a2, a1, :, :, 2], allpwtsgrid[a2, a1, :, :],
                                                   (time_average[a2, a1], frequency_average[a2, a1]))[0].flatten()
                
                result = average_chunks2(visgrid[a2, a1, :, :, :], wtsgrid[a2, a1, :, :, :],
                                         (time_average[a2, a1], frequency_average[a2, a1]))
                cvis[rows], cwts[rows] = result[0].flatten(), result[1].flatten()

                visstart += len_time_chunks[a2, a1]
    
    return cvis, cuvw, cwts, ctime, cfrequency, cpolarisation, ca1, ca2, cintegration_time, cindex


def construct_map(x):
    # Determine the mapping of x to unique x
    unique_x = numpy.unique(x)
    xmap = numpy.zeros(len(x), dtype='int')
    for ind, u in enumerate(unique_x):
        xmap[numpy.where(x == u)] = ind
    return xmap, unique_x


def decoalesce_tbgrid_vis(vshape, cvis, cindex):
    """Decoalesce data using Time-Baseline
    
    We use the index into the coalesced data. For every output row, this gives the
    corresponding row in the coalesced data.

    :param vshape: Shape of template visibility data
    :param cvis: Coalesced visibility values
    :param cindex: Index array from coalescence
    :returns: uncoalesced vis
    """
    dvis = numpy.zeros(vshape, dtype='complex')
    for ind in range(vshape[0]):
        dvis[ind] = cvis[cindex[ind]]

    return dvis
