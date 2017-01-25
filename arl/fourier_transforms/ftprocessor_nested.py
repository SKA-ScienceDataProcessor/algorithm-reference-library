# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing. These are built on top of the core
functions in arl.fourier_transforms.
"""
import multiprocessing

import pymp

from scipy.interpolate import griddata

from arl.fourier_transforms.ftprocessor_base import *
from arl.image.operations import qa_image
from arl.image.iterators import *
from arl.visibility.iterators import *
from arl.visibility.operations import create_visibility_from_rows
from arl.visibility.compress import compress_visibility, decompress_visibility

log = logging.getLogger(__name__)


def invert_nested(vis, im, dopsf=False, invert=invert_2d, **kwargs):
    """Divide vis into inner and outer, image separately, and add

    Cut down on processing by gridding the inner (dense) part of Fourier plane with box gridding,
    and the outer (sparse) part by more accurate PSWF gridding.
    """
    
    # Partition the visibility data
    boundary = get_parameter(kwargs, "inner", 0.125)
    visr = numpy.sqrt(vis.u * vis.u + vis.v * vis.v)
    uvmax = numpy.max(visr)
    inner_rows = (numpy.abs(visr) < boundary * uvmax)
    outer_rows = (numpy.abs(visr) >= boundary * uvmax)
    
    inner_vis = create_visibility_from_rows(vis, inner_rows)
    outer_vis = create_visibility_from_rows(vis, outer_rows)
    log.debug("Split into inner (%d rows) and outer (%d rows)" % (numpy.sum(inner_rows), numpy.sum(outer_rows)))
    
    # Make the contribution to the full scale image from the outer baseline data
    outer_params = copy.copy(kwargs)
    outer_im = create_image_from_visibility(outer_vis, **outer_params)
    outer_result, outer_sumwt = invert(outer_vis, outer_im, dopsf, **outer_params)
    outer_result = normalize_sumwt(outer_result, outer_sumwt)
    print("Outer ", qa_image(outer_result))
    
    # Compress the inner baselines by averaging onto a grid with small cellsize
    cellsize = abs(numpy.pi * im.wcs.wcs.cdelt[0] / 180.0)
    inner_params = copy.copy(kwargs)
    inner_params['cellsize'] = cellsize / boundary
    
    inner_im = create_image_from_visibility(inner_vis, **inner_params)
    inner_vis_compressed = compress_visibility(inner_vis, inner_im, **inner_params)
    
    # Now make the contribution to the full scale image from the compressed data
    inner_result, inner_sumwt = invert(inner_vis_compressed, outer_im, dopsf, **outer_params)
    inner_result = normalize_sumwt(inner_result, inner_sumwt)
    print("Inner ", qa_image(inner_result))

    result = create_image_from_visibility(outer_vis, **outer_params)
    result_sumwt = inner_sumwt + outer_sumwt
    nchan = result.data.shape[0]
    npol = result.data.shape[1]
    for chan in range(nchan):
        for pol in range(npol):
            result.data[chan, pol, ...] = (inner_sumwt[chan, pol] * inner_result.data[chan, pol, ...] +
                                           outer_sumwt[chan, pol] * outer_result.data[chan, pol, ...])
    
    return result, result_sumwt