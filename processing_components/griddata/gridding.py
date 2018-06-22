""" Imaging is based on used of the FFT to perform Fourier transforms efficiently. Since the observed visibility data_models
do not arrive naturally on grid points, the sampled points are resampled on the FFT grid using a convolution function to
smear out the sample points. The resulting grid points are then FFT'ed. The result can be corrected for the gridding
convolution function by division in the image plane of the transform.

This module contains functions for performing the gridding process and the inverse degridding process.

The GridData data model is used to hold the specification of the desired result.
"""

import logging


import numpy

log = logging.getLogger(__name__)


def convolution_mapping(vis, griddata, cf_griddata, channel_tolerance=1e-8):
    """Find the mappings between visibility, griddata, and convolution function
    
    :param vis:
    :param griddata:
    :param cf_griddata:
    :return:
    """
    # Find u, v in pixels
    u_pixel, v_pixel = griddata.grid_wcs.sub([1, 2]).wcs_world2pix(vis.uvw[:, 0], vis.uvw[:, 1], 0)
    # Find the nearest grid point
    u_grid, v_grid = numpy.round(u_pixel).astype('int'), numpy.round(v_pixel).astype('int')
    # Find the fractional part
    u_fraction, v_fraction = u_pixel - u_grid, v_pixel - v_grid
    assert numpy.max(numpy.abs(u_fraction)) <= 0.5
    assert numpy.max(numpy.abs(v_fraction)) <= 0.5
    
    w_pixel = griddata.grid_wcs.sub([3]).wcs_world2pix(vis.uvw[:, 2], 0)[0]
    # Find the nearest grid point
    w_grid = numpy.round(w_pixel).astype('int')
    # Find the fractional part
    w_fraction = w_pixel - w_grid
    
    print("UV", u_pixel[0:10], v_pixel[0:10])
    print("UV", u_grid[0:10], v_grid[0:10])
    print("UV", u_fraction[0:10], v_fraction[0:10])
    
    print("W", w_pixel[0:10], w_grid[0:10], w_fraction[0:10])
    
    freq_pixel = griddata.grid_wcs.sub([5]).wcs_world2pix(vis.frequency, 0)[0]
    # Find the nearest grid point
    freq_grid = numpy.round(freq_pixel).astype('int')
    # Find the fractional part
    freq_fraction = freq_pixel - freq_grid
    if numpy.max(numpy.abs(freq_fraction)) > channel_tolerance:
        log.warning("convolution_mapping: alignment of visibility and image grids exceeds tolerance %s" %
                    (str(numpy.max(max.abs(freq_fraction))), str(channel_tolerance)))
    
    print("FREQ", freq_pixel[0:10], freq_grid[0:10], freq_fraction[0:10])
    
    return u_grid, u_fraction, v_grid, v_grid, w_grid, w_fraction, freq_grid



def grid_visibility_to_griddata(vis, griddata, cf_griddata, **kwargs):
    """Grid Visibility onto a GridData

    :param vis: Visibility to be gridded
    :param griddata: GridData
    :param cf: Convolution function (as GridData)
    :param kwargs:
    :return: GridData
    """
    nchan, npol, _, ny, nx = griddata.data.shape
    
    sumwt = numpy.zeros([nchan, npol])
    
    u_grid, u_fraction, v_grid, v_fraction, w_grid, w_fraction, freq_grid = convolution_mapping(vis, griddata,
                                                                                                cf_griddata)
    
    _, _, _, gy, gx = cf_griddata.shape
    
    coords = zip(vis.vis, vis.weight,
                 freq_grid,
                 u_grid - gx // 2, u_fraction,
                 v_grid - gy // 2, v_fraction,
                 w_grid, freq_grid)
    
    griddata.data[...] = 0.0
    
    for pol in range(npol):
        for v, vwt, chan, xx, yy, xxf, yyf, zz, ff in coords:
            griddata.data[chan, pol, zz, yy: yy + gy, xx: xx + gx] += \
                cf_griddata.data[chan, pol, yyf, xxf, zz, :, :] * v[pol] * vwt
            sumwt[chan, pol] += vwt
    
    return griddata, sumwt


def degrid_visibility_from_griddata(vis, griddata, cf_griddata, **kwargs):
    """Degrid Visibility from a GridData

    :param vis: Visibility to be degridded
    :param griddata: GridData containing image
    :param cf: Convolution function (as GridData)
    :param kwargs:
    :return: Visibility
    """
    return vis
