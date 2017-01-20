# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing. These are built on top of the core
functions in arl.fourier_transforms.
"""

from arl.fourier_transforms.convolutional_gridding import frac_coord
from arl.fourier_transforms.fft_support import *
from arl.fourier_transforms.ftprocessor import *

log = logging.getLogger(__name__)


def compress_visibility(vis, im, **kwargs):
    """ Compress the visibility data using a grid

    Compress by gridding the visibilities onto a fine grid and then extracting the visibilities, weights and uvw
    from the grid. The maximum number of rows in the output visibility is the same number as the number of pixels
    in each polarisation-frequency plane i.e. nx, ny

    :param vis:
    :param im:
    :param compression: Only "uvgrid" is currently supported
    :returns: Derived Visibility
    """
    nchan, npol, ny, nx = im.data.shape
    vmap, gcf, kernel_type, kernelname, kernel, padding, oversampling, support, \
    cellsize, fov, uvscale = get_ftprocessor_params(vis, im, **kwargs)
    
    assert nx == im.data.shape[3], "Discrepancy between npixel and size of model image"
    assert ny == im.data.shape[2], "Discrepancy between npixel and size of model image"
    
    compression = get_parameter(kwargs, "compression", "uvgrid")
    if compression == 'uvgrid':
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
    
    else:
        log.error("Unknown visibility compression algorithm %s" % compression)
    return compressed_vis


def decompress_visibility(vis, template_vis, im, **kwargs):
    """ Decompress the visibilities from a gridded set to the original values (opposite of compress_visibility)

    :param vis: (Compressed visibility
    :param template_vis: Template visibility to be filled in
    :param im: Image specifying coordinates of image (must be consistent with vis)
    :param compression: Only "uvgrid" is currently supported
    :returns: New visibility with vis and weight columns overwritten
    """
    nchan, npol, ny, nx = im.data.shape
    vmap, gcf, kernel_type, kernelname, kernel, padding, oversampling, support, \
    cellsize, fov, uvscale = get_ftprocessor_params(vis, im, **kwargs)
    
    assert nx == im.data.shape[3], "Discrepancy between npixel and size of model image"
    assert ny == im.data.shape[2], "Discrepancy between npixel and size of model image"
    
    compression = get_parameter(kwargs, "compression", "uvgrid")
    decomp_vis = template_vis
    if compression == 'uvgrid':
        log.info('decompress_visibility: Created new Visibility for decompressed data')
        log.info('decompress_visibility: Decompressing %d visibility rows (%d channels) into %d '
                 'rows (%d channels)' %
                 (vis.nvis, nchan, template_vis.nvis, len(template_vis.frequency)))
        decomp_vis = copy.deepcopy(template_vis)
        decomp_vis.data['vis'], decomp_vis.data['weight'] = \
            decompress_uvgrid_vis(im.data.shape,
                                  template_vis.data['vis'],
                                  template_vis.data['weight'],
                                  template_vis.data['uvw'],
                                  vis.data['vis'],
                                  vis.data['weight'],
                                  vis.data['uvw'],
                                  uvscale,
                                  vmap)
    
    else:
        log.error("Unknown visibility compression algorithm %s" % compression)
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

    visgrid[wtsgrid>0.0] /= wtsgrid[wtsgrid>0.0]
    visgrid[wtsgrid<=0.0] = 0.0
    
    # These just need a reshape
    nvis = nx * ny
    cvis        = visgrid.reshape(nvis, inchan, inpol)
    cvisweights = wtsgrid.reshape(nvis, inchan, inpol)
    
    # Need to convert back to metres at the first frequency
    cu = (numpy.arange(nx) - nx // 2) / (nx * uvscale[0, 0])
    cv = (numpy.arange(ny) - ny // 2) / (ny * uvscale[1, 0])
    cuvw = numpy.zeros([nx * ny, 3])
    for v in range(len(cv)):
        for u in range(len(cu)):
            cuvw[u+len(cu)*v, 0] = cu[u]
            cuvw[u+len(cu)*v, 1] = cv[v]
            
    # Construct row mask
    rowmask = numpy.where(cvisweights > 0)[0]
    return cvis[rowmask,...], cuvw[rowmask,...], cvisweights[rowmask,...], rowmask


def decompress_uvgrid_vis(shape, tvis, twts, tuvw, vis, visweights, uvw, uvscale, vmap):
    """Decompress data using one of a number of algorithms

    :param shape:
    :param tvis: Template visibility
    :param twts: Template weights
    :param tuvw: Template UVW positions
    :param vis: Visibility values
    :param visweights: Visibility weights
    :param uvscale: Scaling for each axis (u,v) for each channel
    """
    nchan, npol, ny, nx = shape
    rshape = ny, nx, nchan, npol
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

    tvis[...] = 0.0
    twts[...] = 0.0
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

def compress_tbgrid_vis(shape, vis, time, antenna1, antenna2, uvw, visweights):
    """Compress data by gridding onto a time baseline grid

    :param shape: Shape of grid to be used => shape of output visibilities
    :param uv: Input UVW positions
    :param uvscale: Scaling for each axis (u,v) for each channel
    :param vis: Visibility values
    :param visweights: Visibility weights
    :returns: vis, uvw, visweights
    """
    # Find the maximum possible baseline and then scale to this.
    uvmax = numpy.max(vis.u**2+vis.v**2+vis**2)
    nchan, npol, ny, nx = shape
    
    ntimes = len(numpy.unique(time))
    nant = numpy.max(antenna2)
    
    log.info('compress_tbgrid_vis: Compressing %d unique times and %d baselines' % (ntimes, nbaselines))
    
    visshape = ntimes, nant, nant, nchan, npol
    visgrid = numpy.zeros(visshape, dtype='complex')
    wtsgrid = numpy.zeros(visshape)

    uvwshape = ntimes, nant, nant, 3
    uvwgrid = numpy.zeros(uvwshape)
    
    timeoffset = numpy.min(time)
    timescale = 1.0 / (numpy.max(time) - timeoffset)
    time_integration = timescale / (len(time) - 1)
    log.info('compress_tbgrid_vis: Integration time seems to be *.2f (seconds)')

    # Add all visibility points to a grid
    tindex = (numpy.round(timescale * (time-timeoffset))).astype('int')
    coords = tindex, antenna1, antenna2
    for chan in range(nchan):
        for pol in range(npol):
            vs = vis[..., chan, pol]
            wts = visweights[..., chan, pol]
            for v, wt, t, a1, a2 in zip(vs, wts, *coords):
                visgrid[t, a1, a2, chan, pol] = v
                wtsgrid[t, a1, a2, chan, pol] = wt
                
    # Calculate the scaled integration time making it the same for all times for this baseline
    t_average = numpy.zeros([nant, nant])
    for t in range(ntimes):
        for a2 in range(nant):
            uvdist = numpy.max(numpy.sqrt(uvwgrid[t, :, a2, 0]**2 + uvwgrid[t, :, a2, 1]**2))
            t_average[:, a2] = 1 + int(round((uvmax / uvdist)))
            
    log.info('compress_tbgrid_vis: Number of integrations to average: %s' % t_average)
    
    visgrid[wtsgrid > 0.0] /= wtsgrid[wtsgrid > 0.0]
    visgrid[wtsgrid <= 0.0] = 0.0
    
    # Now average and extract. Since we don't know how many visibilities will be produced, we
    # insert into a list and then convert to numpy arrays
    lvis = []
    luvw = []
    ltime = []
    lvisweights = []
    la1 = []
    la2 = []
    
    cvis = numpy.array(lvis)
    cuvw = numpy.array(luvw)
    ctime = numpy.array(ltime)
    cvisweights = numpy.array(lvisweights)
    ca1 = numpy.array(la1)
    ca2 = numpy.array(la2)

    rowmask = numpy.where(cvisweights > 0)[0]
    return cvis[rowmask, ...], cuvw[rowmask, ...], ctime[rowmask], cvisweights[rowmask, ...], ca1[rowmask], ca1[rowmask]