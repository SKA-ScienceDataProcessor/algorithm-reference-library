# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing. These are built on top of the core
functions in arl.fourier_transforms.
"""

from arl.fourier_transforms.convolutional_gridding import frac_coord
from arl.fourier_transforms.ftprocessor_base import *
from arl.visibility.operations import vis_summary

log = logging.getLogger(__name__)


def compress_visibility(vis, im=None, **kwargs):
    """ Compress the visibility data using a grid

    Compress by gridding the visibilities onto a fine grid and then extracting the visibilities, weights and uvw
    from the grid. The maximum number of rows in the output visibility is the same number as the number of pixels
    in each polarisation-frequency plane i.e. nx, ny

    :param vis: Visibility to be compressed
    :param im: Only needed for uv compression
    :param compression:  "uv" | "tb" currently supported
    :returns: Compressed visibility
    """
    compression = get_parameter(kwargs, "compression", "uvgrid")
    compressed_vis = None
    if compression == 'uv':
        assert im, "uv compression needs an image"
        nchan, npol, ny, nx = im.data.shape
        vmap, gcf, kernel_type, kernelname, kernel, padding, oversampling, support, \
        cellsize, fov, uvscale = get_ftprocessor_params(vis, im, **kwargs)

        assert nx == im.data.shape[3], "Discrepancy between npixel and size of model image"
        assert ny == im.data.shape[2], "Discrepancy between npixel and size of model image"

        cvis, cuvw, cvisweights, rowmask = compress_uvgrid_vis(im.data.shape, vis.data['vis'], vis.data['uvw'], uvscale,
                                                               vis.data['weight'], vmap)
        newfrequency = numpy.array(im.wcs.wcs.crval[3])
        nrows = cvis.shape[0]
        ca1 = numpy.zeros([nrows])
        ca2 = numpy.zeros([nrows])
        cimwt = numpy.ones(cvis.shape)
        ctime = numpy.ones(nrows) * numpy.average(vis.data['time'])
        compressed_vis = Visibility(uvw=cuvw, time=ctime, frequency=newfrequency, phasecentre=vis.phasecentre,
                                    antenna1=ca1, antenna2=ca2, vis=cvis, weight=cvisweights,
                                    imaging_weight=cimwt, configuration=vis.configuration)
        assert rowmask.shape[0] == compressed_vis.nvis, "Discrepancy in number of rows: rowmask %s, nvis %s" \
                                                        % (rowmask.shape[0], compressed_vis.nvis)
        log.info('compress_visibility: Created new Visibility for compressed data')
        log.info('compress_visibility: Compressed %d visibility rows (%d channels) into %d rows (%d channels)' %
                 (vis.nvis, len(vis.frequency), compressed_vis.nvis, nchan))
    
    elif compression == 'tb':
        compression_factor = get_parameter(kwargs, "compression_factor", 1.0)
        max_compression    = get_parameter(kwargs, "max_compression", 10)
        cvis, cuvw, ctime, cvisweights, ca1, ca2, cintegration_time, cindex = \
            compress_tbgrid_vis(vis.data['vis'], vis.data['time'], vis.data['antenna1'], vis.data['antenna2'],
                                vis.data['uvw'], vis.data['weight'], vis.data['integration_time'],
                                max_compression=max_compression, compression_factor=compression_factor)
        
        newfrequency = vis.frequency
        cimwt = numpy.ones(cvis.shape)
        nrows = cvis.shape[0]
        cintegration_time = numpy.ones(nrows)
        compressed_vis = Visibility(uvw=cuvw, time=ctime, frequency=newfrequency, phasecentre=vis.phasecentre,
                                    antenna1=ca1, antenna2=ca2, vis=cvis, weight=cvisweights,
                                    imaging_weight=cimwt, configuration=vis.configuration,
                                    integration_time=cintegration_time)
        
        log.info('compress_visibility: Created new Visibility for compressed data, compression factor = %.3f' % (
            compression_factor))
        log.info('compress_visibility: Compressed %d visibility rows (%d channels) into %d rows (%d channels)' %
                 (vis.nvis, len(vis.frequency), compressed_vis.nvis, len(vis.frequency)))
    else:
        log.error("Unknown visibility compression algorithm %s" % compression)
        
    log.info('compress_visibility: Original %s, compressed %s' % (vis_summary(vis), vis_summary(compressed_vis)))
    return compressed_vis, cindex


def decompress_visibility(vis, template_vis, im=None, cindex=None, **kwargs):
    """ Decompress the visibilities from a gridded set to the original values (opposite of compress_visibility)

    :param vis: (Compressed visibility
    :param template_vis: Template visibility to be filled in
    :param im: Image specifying coordinates of image (must be consistent with vis)
    :param compression: Only "uvgrid" is currently supported
    :returns: New visibility with vis and weight columns overwritten
    """
   
    compression = get_parameter(kwargs, "compression", "uvgrid")
    decomp_vis = template_vis
    if compression == 'uv':
        assert im is not None, "UV decompression needs an image"
        nchan, npol, ny, nx = im.data.shape
        vmap, gcf, kernel_type, kernelname, kernel, padding, oversampling, support, \
        cellsize, fov, uvscale = get_ftprocessor_params(vis, im, **kwargs)
    
        assert nx == im.data.shape[3], "Discrepancy between npixel and size of model image"
        assert ny == im.data.shape[2], "Discrepancy between npixel and size of model image"
    
        log.info('decompress_visibility: Created new Visibility for decompressed data')
        log.info('decompress_visibility: Decompressing %d visibility rows (%d channels) into %d '
                 'rows (%d channels)' %
                 (vis.nvis, nchan, template_vis.nvis, len(template_vis.frequency)))
        decomp_vis = copy.deepcopy(template_vis)
        decomp_vis.data['vis'], decomp_vis.data['weight'] = \
            decompress_uvgrid_vis(im.data.shape,
                                  template_vis.data['uvw'],
                                  vis.data['vis'],
                                  vis.data['weight'],
                                  vis.data['uvw'],
                                  uvscale,
                                  vmap)
    elif compression == 'tb':
        log.info('decompress_visibility: Created new Visibility for decompressed data')
        log.info('decompress_visibility: Decompressing %d visibility rows (%d channels) into %d '
                 'rows (%d channels)' %
                 (vis.nvis, len(template_vis.frequency), template_vis.nvis, len(template_vis.frequency)))
        decomp_vis = copy.deepcopy(template_vis)
        decomp_vis.data['vis'] = \
            decompress_tbgrid_vis(template_vis.data['vis'].shape,
                                  vis.data['vis'],
                                  cindex)

    else:
        log.error("Unknown visibility compression algorithm %s" % compression)
    log.info('compress_visibility: Original %s, decompressed %s' % (vis_summary(vis), vis_summary(decomp_vis)))
    return decomp_vis


def compress_uvgrid_vis(shape, vis, uvw, uvscale, visweights, vmap):
    """Compress data by gridding with box and then converting the grid to visibilities, uvw, and weights
    
    :param shape: Shape of grid to be used => shape of output visibilities
    :param uv: Input UVW positions
    :param uvscale: Scaling for each axis (u,v) for each channel
    :param vis: Visibility values
    :param visweights: Visibility weights
    :returns: vis, uvw, visweights
    """
    inchan, inpol, ny, nx = shape
    nvis, vnchan, vnpol = vis.shape
    rshape = ny, nx, inchan, inpol
    visgrid = numpy.zeros(rshape, dtype='complex')
    wtsgrid = numpy.zeros(rshape)
    # Add all visibility points to a float grid
    for vchan in range(vnchan):
        ichan = vmap(vchan)
        y, _ = frac_coord(ny, 1, uvscale[1, vchan] * uvw[..., 1])
        x, _ = frac_coord(nx, 1, uvscale[0, vchan] * uvw[..., 0])
        coords = x, y
        for vpol in range(vnpol):
            vs = vis[..., vchan, vpol]
            wts = visweights[..., vchan, vpol]
            for v, wt, x, y, in zip(vs, wts, *coords):
                # Note that this ordering is different from that in other gridding
                # functions because we want to do a reshape to get the result into
                # the correct order
                visgrid[y, x, ichan, vpol] += v
                wtsgrid[y, x, ichan, vpol] += wt
    
    visgrid[wtsgrid > 0.0] /= wtsgrid[wtsgrid > 0.0]
    visgrid[wtsgrid <= 0.0] = 0.0
    
    # These just need a reshape
    nvis = nx * ny
    cvis = visgrid.reshape(nvis, inchan, inpol)
    cvisweights = wtsgrid.reshape(nvis, inchan, inpol)
    
    # Need to convert back to metres at the first frequency
    cu = (numpy.arange(nx) - nx // 2) / (nx * uvscale[0, 0])
    cv = (numpy.arange(ny) - ny // 2) / (ny * uvscale[1, 0])
    cuvw = numpy.zeros([nx * ny, 3])
    for v in range(len(cv)):
        for u in range(len(cu)):
            cuvw[u + len(cu) * v, 0] = cu[u]
            cuvw[u + len(cu) * v, 1] = cv[v]
    
    # Construct row mask
    rowmask = numpy.where(cvisweights > 0)[0]
    return cvis[rowmask, ...], cuvw[rowmask, ...], cvisweights[rowmask, ...], rowmask


def decompress_uvgrid_vis(shape, tuvw, vis, visweights, uvw, uvscale, vmap):
    """Decompress data using one of a number of algorithms

    :param tvis: Template visibility
    :param twts: Template weights
    :param tuvw: Template UVW positions
    :param vis: Visibility values
    :param visweights: Visibility weights
    :param uvscale: Scaling for each axis (u,v) for each channel
    """

    inchan, inpol, ny, nx = shape
    rshape = ny, nx, inchan, inpol
    nvis, vnchan, vnpol = vis.shape
    visgrid = numpy.zeros(rshape, dtype='complex')
    wtsgrid = numpy.zeros(rshape)
    # First rebuild the full grid. We could have kept it cached.
    for vchan in range(vnchan):
        ichan = vmap(vchan)
        yy, _ = frac_coord(ny, 1, uvscale[1, vchan] * uvw[..., 1])
        xx, _ = frac_coord(nx, 1, uvscale[0, vchan] * uvw[..., 0])
        coords = xx, yy
        for vpol in range(vnpol):
            vs = vis[..., vchan, vpol]
            wts = visweights[..., vchan, vpol]
            for v, wt, x, y, in zip(vs, wts, *coords):
                # Note that this ordering is different from that in other gridding
                # functions because we want to do a reshape to get the result into
                # the correct order
                # Note also that this is a one-to-one mapping
                visgrid[y, x, ichan, vpol] = v
                wtsgrid[y, x, ichan, vpol] = wt

 
    dvshape = tuvw.shape[0], vnchan, vnpol
    tvis = numpy.zeros(dvshape, dtype='complex')
    twts = numpy.zeros(dvshape)
    
    for vchan in range(vnchan):
        ichan = vmap(vchan)
        y, _ = frac_coord(ny, 1, uvscale[1, vchan] * tuvw[..., 1])
        x, _ = frac_coord(nx, 1, uvscale[0, vchan] * tuvw[..., 0])
        for vpol in range(vnpol):
            # Note that this ordering is different from that in other gridding
            # functions because we want to do a reshape to get the result into
            # the correct order
            # Note also that this is a one-to-many mapping so the results
            # can be redundant
            tvis[..., vchan, vpol] = visgrid[y, x, ichan, vpol]
            twts[..., vchan, vpol] = wtsgrid[y, x, ichan, vpol]
    
    return tvis, twts


def average_chunks(arr, wts, chunksize):
    """ Average the array arr with weights by chunks
    """
    if chunksize > len(arr):
        return arr, wts
    elif chunksize <= 1:
        return arr, wts
    
    nchunks, rem = divmod(len(arr), chunksize)
    if rem:
        nchunks += 1
    
    chunks = numpy.zeros(nchunks, dtype=arr.dtype)
    weights = numpy.zeros(nchunks, dtype=wts.dtype)
    
    ichunk = 0
    for i in range(0, len(arr), chunksize):
        ind = slice(i, min(i + chunksize, len(arr)))
        weights[ichunk] = numpy.sum(wts[ind])
        chunks[ichunk] = numpy.sum(wts[ind] * arr[ind])
        ichunk += 1
        
    chunks[weights>0.0] = chunks[weights>0.0] / weights[weights>0.0]
    
    return chunks, weights


def compress_tbgrid_vis(vis, time, antenna1, antenna2, uvw, visweights, integration_time, max_compression=10,
                        compression_factor=1.0):
    """Compress data by gridding onto a time baseline grid

    :param shape: Shape of grid to be used => shape of output visibilities
    :param uv: Input UVW positions
    :param uvscale: Scaling for each axis (u,v) for each channel
    :param vis: Visibility values
    :param visweights: Visibility weights
    :param max_compression: Maximum number of elements to average
    :param compression_factor: Boost factor for compression > 1 implies more averaging
    :returns: vis, uvw, visweights
    """
    # Find the maximum possible baseline and then scale to this.
    uvmax = numpy.sqrt(numpy.max(uvw[:, 0] ** 2 + uvw[:, 1] ** 2 + uvw[:, 1] ** 2))
    
    nvis, vnchan, vnpol = vis.shape
    
    utimes = numpy.unique(time)
    ntimes = len(utimes)
    nant = numpy.max(antenna2) + 1
    nbaselines = nant * (nant + 1)
    
    log.info('compress_tbgrid_vis: Compressing %d unique times and %d baselines' % (ntimes, nbaselines))
    
    visshape = nant, nant, ntimes, vnchan, vnpol
    visgrid = numpy.zeros(visshape, dtype='complex')
    wtsgrid = numpy.zeros(visshape)
    
    uvwshape = nant, nant, ntimes, 3
    uvwgrid = numpy.zeros(uvwshape)
    timeshape = nant, nant, ntimes
    timegrid = numpy.zeros(timeshape)
    integration_time_grid = numpy.zeros(timeshape)
    
    # Determine the mapping of times to unique times
    timemap = numpy.zeros(len(time), dtype='int')
    for ind, u in enumerate(utimes):
        timemap[numpy.where(time == u)] = ind
    
    rows = range(nvis)
    coords = rows, timemap, time, integration_time, antenna2, antenna1
  
    # Assume that a2 > a1. We rearrange the visibility data to be per time per baseline.
    # i.e. antenna2, antenna1, time, chan, pol instead of
    #      time, antenna2, antenna2, chan, pol
    for chan in range(vnchan):
        for pol in range(vnpol):
            vs = vis[..., chan, pol]
            wts = visweights[..., chan, pol]
            for v, wt, row, tm, tg, it, a2, a1 in zip(vs, wts, *coords):
                visgrid[a2, a1, tm, chan, pol] = v
                wtsgrid[a2, a1, tm, chan, pol] = wt

    # To make life easier, we will keep the mapping between input row numbers
    rowgrid = numpy.zeros(timeshape, dtype='int')
    
    coords = rows, timemap, time, integration_time, antenna2, antenna1, uvw[:, 0], uvw[:, 1], uvw[:, 2]
    for row, tm, tg, it, a2, a1, uu, vv, ww in zip(*coords):
        uvwgrid[a2, a1, tm, 0] = uu
        uvwgrid[a2, a1, tm, 1] = vv
        uvwgrid[a2, a1, tm, 2] = ww
        timegrid[a2, a1, tm] = tg
        integration_time_grid[a2, a1, tm] = it
        rowgrid[a2, a1, tm] = row

    visgrid[wtsgrid > 0.0] /= wtsgrid[wtsgrid > 0.0]
    visgrid[wtsgrid <= 0.0] = 0.0
    
    time_integration = utimes[1] - utimes[0]
    log.info('compress_tbgrid_vis: Time step between integrations seems to be %.2f (seconds)' % time_integration)
    
    # Calculate the scaled integration time making it the same for all times for this baseline
    sample_width = numpy.zeros([nant, nant], dtype='int')
    ua2 = numpy.unique(antenna2)
    ua1 = numpy.unique(antenna1)
    for a2 in ua2:
        for a1 in ua1:
            if (a1 < a2) & (wtsgrid[a2, a1, ...].any() > 0.0):
                uvdist = numpy.max(numpy.sqrt(uvwgrid[a2, a1, :, 0] ** 2 + uvwgrid[a2, a1, :, 1] ** 2))
                if uvdist > 0.0:
                    sample_width[a2, a1] = min(max_compression, max(1, int(round((compression_factor * uvmax / uvdist)))))
    
    # See how many time chunks we need for each baseline. To do this we use the same averaging that
    # we will use later for the actual data. This tells us the number of chunks required for each baseline.
    cnvis = 0
    len_time_chunks = numpy.ones([nant, nant], dtype='int')
    for a2 in ua2:
        for a1 in ua1:
            if (a1 < a2) & (sample_width[a2,a1] > 0):
                time_chunks, wts = average_chunks(timegrid[a2, a1, :], wtsgrid[a2, a1, :, 0, 0], sample_width[a2, a1])
                len_time_chunks[a2, a1] = len(time_chunks)
            cnvis += len_time_chunks[a2, a1]
   
    # Now we know enough to define the output compressed arrays
    ctime = numpy.zeros([cnvis])
    cvis = numpy.zeros([cnvis, vnchan, vnpol], dtype='complex')
    cwts = numpy.zeros([cnvis, vnchan, vnpol], dtype='float')
    cuvw = numpy.zeros([cnvis, 3])
    ca1 = numpy.zeros([cnvis], dtype='int')
    ca2 = numpy.zeros([cnvis], dtype='int')
    cintegration_time = numpy.zeros([cnvis])

    # Now go through, chunking up the various arrays. As written this produces sort order with a2, a1 varying slowest
    cindex = numpy.zeros([nvis], dtype='int')
    visstart = 0
    for a2 in ua2:
        for a1 in ua1:
            if (a1 < a2) & (len_time_chunks[a2,a1] > 0):
                rows = slice(visstart, visstart + len_time_chunks[a2, a1])
                
                cindex[rowgrid[a2,a1,:]] = numpy.array(range(visstart, visstart + len_time_chunks[a2, a1]))

                ca1[rows] = a1
                ca2[rows] = a2
                
                cintegration_time[rows], _ = average_chunks(integration_time_grid[a2, a1, :], wtsgrid[a2, a1, :, 0, 0],
                                                         sample_width[a2, a1])

                ctime[rows], _ = average_chunks(timegrid[a2, a1, :], wtsgrid[a2, a1, :, 0, 0], sample_width[a2, a1])
                
                cuvw[rows, 0], _ = average_chunks(uvwgrid[a2, a1, :, 0], wtsgrid[a2, a1, :, 0, 0], sample_width[a2, a1])
                cuvw[rows, 1], _ = average_chunks(uvwgrid[a2, a1, :, 1], wtsgrid[a2, a1, :, 0, 0], sample_width[a2, a1])
                cuvw[rows, 2], _ = average_chunks(uvwgrid[a2, a1, :, 2], wtsgrid[a2, a1, :, 0, 0], sample_width[a2, a1])
                
                for chan in range(vnchan):
                    for pol in range(vnpol):
                        cvis[rows, chan, pol], cwts[rows, chan, pol] = \
                            average_chunks(visgrid[a2, a1, :, chan, pol], wtsgrid[a2, a1, :, chan, pol],
                                           sample_width[a2, a1])
    
                visstart += len_time_chunks[a2, a1]
    
    return cvis, cuvw, ctime, cwts, ca1, ca2, cintegration_time, cindex


def decompress_tbgrid_vis(vshape, cvis, cindex):
    """Decompress data using Time-Baseline
    
    We use the index into the compressed data. For every output row, this gives the
    corresponding row in the compressed data.

    :param vshape: Shape of template visibility data
    :param cvis: Compressed visibility values
    :param cindex: Index array from compression
    :returns: uncompressed vis
    """
    dvis = numpy.zeros(vshape, dtype='complex')
    for ind in range(vshape[0]):
        dvis[ind,:,:] = cvis[cindex[ind],:,:]
    
    return dvis
