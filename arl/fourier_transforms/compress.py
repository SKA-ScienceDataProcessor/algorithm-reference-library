# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing. These are built on top of the core
functions in arl.fourier_transforms.
"""

from astropy import units as units
from astropy import wcs
from astropy.constants import c
from astropy.wcs.utils import pixel_to_skycoord

from arl.data.data_models import *
from arl.fourier_transforms.convolutional_gridding import frac_coord
from arl.fourier_transforms.ftprocessor import *
from arl.fourier_transforms.fft_support import *

log = logging.getLogger(__name__)

def compress_visibility(vis, im, **kwargs):
    """ Compress the visibility data using a grid

    :param vis:
    :param im:
    :returns: visibility with imaging_weights column added and filled
    """
    nchan, npol, ny, nx, shape, gcf, kernel_type, kernelname, kernel,padding, oversampling, support, cellsize, \
    fov, uvscale = get_ftprocessor_params(vis, im, **kwargs)

    assert nx==im.data.shape[3], "Discrepancy between npixel and size of model image"
    assert ny==im.data.shape[2], "Discrepancy between npixel and size of model image"

    compression = get_parameter(kwargs, "compression", "uvgrid")
    if compression == 'uvgrid':
        cvis, cuvw, cvisweights= compress_grid_vis(im.data.shape, vis.data['vis'], vis.data['uvw'], uvscale,
                                                   vis.data['weight'])
        nrows = cvis.shape[0]
        ca1 = numpy.zeros([nrows])
        ca2 = numpy.zeros([nrows])
        cimwt = numpy.ones(cvis.shape)
        ctime = numpy.ones(nrows) * numpy.average(vis.data['time'])
        compressed_vis = Visibility(uvw=cuvw, time=ctime, frequency=vis.frequency, phasecentre=vis.phasecentre,
                                    antenna1=ca1, antenna2=ca2, vis=cvis, weight=cvisweights,
                                    imaging_weight=cimwt, configuration=vis.configuration)
        log.info('compress_visibility: Compressed %d visibility rows into %d rows on a regular grid' %
                 (vis.nvis, compressed_vis.nvis))

    else:
        log.error("Unknown visibility compression algorithm %s" % compression)
    return compressed_vis

def compress_grid_vis(shape, vis, uv, uvscale, visweights):
    """Compress data using one of a number of algorithms

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param shape:
    :param uv: UVW positions
    :param uvscale: Scaling for each axis (u,v) for each channel
    :param vis: Visibility values
    :param visweights: Visibility weights
    :param weighting: Weighting algorithm (natural|uniform) (uniform)
    """
    nchan, npol, ny, nx = shape
    
    rshape = ny, nx, nchan, npol
    visgrid = numpy.zeros(rshape, dtype='complex')
    wtsgrid = numpy.zeros(rshape)
    # Add all visibility points to a float grid
    for chan in range(1):
        y, _ = frac_coord(ny, 1, uvscale[1, chan] * uv[..., 1])
        x, _ = frac_coord(nx, 1, uvscale[0, chan] * uv[..., 0])
        coords = x, y
        for pol in range(npol):
            vs = vis[..., chan, pol]
            wts = visweights[..., chan, pol]
            for v, wt, x, y, in zip(vs, wts, *coords):
                # Note that this ordering is different from that in other gridding
                # functions because we want to do a reshape to get the result into
                # the correct order
                visgrid[y, x, chan, pol] += v
                wtsgrid[y, x, chan, pol] += wt

    visgrid[wtsgrid>0.0] /= wtsgrid[wtsgrid>0.0]
    visgrid[wtsgrid<=0.0] = 0.0
    
    # These just need a reshape
    nvis = nx * ny
    cvis = visgrid.reshape(nvis, nchan, npol)
    cvisweights = wtsgrid.reshape(nvis, nchan, npol)
    
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
    return cvis[rowmask,...], cuvw[rowmask,...], cvisweights[rowmask,...]

