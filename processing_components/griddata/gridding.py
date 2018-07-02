""" Imaging is based on used of the FFT to perform Fourier transforms efficiently. Since the observed visibility data_models
do not arrive naturally on grid points, the sampled points are resampled on the FFT grid using a convolution function to
smear out the sample points. The resulting grid points are then FFT'ed. The result can be corrected for the gridding
convolution function by division in the image plane of the transform.

This module contains functions for performing the gridding process and the inverse degridding process.

The GridData data model is used to hold the specification of the desired result.
"""

import logging

import numpy
import numpy.testing

from libs.image.operations import fft, create_image_from_array

log = logging.getLogger(__name__)


def convolution_mapping(vis, griddata, cf, channel_tolerance=1e-8):
    """Find the mappings between visibility, griddata, and convolution function
    
    :param vis:
    :param griddata:
    :param cf_griddata:
    :return:
    """
    numpy.testing.assert_almost_equal(griddata.grid_wcs.wcs.cdelt[0], cf.grid_wcs.wcs.cdelt[0], 7)
    numpy.testing.assert_almost_equal(griddata.grid_wcs.wcs.cdelt[1], cf.grid_wcs.wcs.cdelt[1], 7)
    
    ####### UV mapping
    # We use the wcs's available to do the coordinate conversion
    # Find the nearest grid points
    print(griddata.grid_wcs)
    print(cf.grid_wcs)
    pu_grid, pv_grid = \
        numpy.round(griddata.grid_wcs.sub([1, 2]).wcs_world2pix(vis.uvw[:, 0], vis.uvw[:, 1], 0)).astype('int')
    assert numpy.min(pu_grid) >= 0
    assert numpy.max(pu_grid) < griddata.shape[3]
    assert numpy.min(pv_grid) >= 0
    assert numpy.max(pv_grid) < griddata.shape[4]
    
    # We now have the location of grid points, convert back to uv space and find the remainder (in wavelengths). We
    # then use this to calculate the subsampling indices (DUU, DVV)
    wu_grid, wv_grid = griddata.grid_wcs.sub([1, 2]).wcs_pix2world(pu_grid, pv_grid, 0)
    wu_subsample, wv_subsample = vis.uvw[:, 0] - wu_grid, vis.uvw[:, 1] - wv_grid
    
    # If everything is correct, the fraction in pixels must lie in the range -0.5 to 0.5 so we round it.
    pu_offset, pv_offset = \
        numpy.floor(cf.grid_wcs.sub([3, 4]).wcs_world2pix(wu_subsample, wv_subsample, 0)).astype('int')
    assert numpy.min(pu_offset) >= 0, numpy.min(pu_offset)
    assert numpy.max(pu_offset) < cf.shape[3], numpy.max(pu_offset)
    assert numpy.min(pv_offset) >= 0, numpy.min(pv_offset)
    assert numpy.max(pv_offset) < cf.shape[4], numpy.max(pv_offset)
    
    ###### W mapping
    pw_pixel = griddata.grid_wcs.sub([3]).wcs_world2pix(vis.uvw[:, 2], 0)[0]
    # Find the nearest grid point
    pw_grid = numpy.round(pw_pixel).astype('int')
    assert numpy.min(pw_grid) >= 0
    assert numpy.max(pw_grid) < cf.shape[2]
    pw_fraction = pw_pixel - pw_grid
    
    ###### Frequency mapping
    pfreq_pixel = griddata.grid_wcs.sub([5]).wcs_world2pix(vis.frequency, 0)[0]
    # Find the nearest grid point
    pfreq_grid = numpy.round(pfreq_pixel).astype('int')
    pfreq_fraction = pfreq_pixel - pfreq_grid
    if numpy.max(numpy.abs(pfreq_fraction)) > channel_tolerance:
        log.warning("convolution_mapping: alignment of visibility and image grids exceeds tolerance %s" %
                    (str(channel_tolerance)))
    
    ######  TODO: Polarisation mapping
    
    return pu_grid, pu_offset, pv_grid, pv_offset, pw_grid, pw_fraction, pfreq_grid


def grid_visibility_to_griddata(vis, griddata, cf, gcf):
    """Grid Visibility onto a GridData

    :param vis: Visibility to be gridded
    :param griddata: GridData
    :param cf: Convolution function
    :param kwargs:
    :return: GridData
    """
    nchan, npol, nz, oversampling, _, support, _ = cf.shape
    sumwt = numpy.zeros([nchan, npol])
    pu_grid, pu_offset, pv_grid, pv_offset, pw_grid, pw_fraction, pfreq_grid = convolution_mapping(vis, griddata, cf)
    _, _, _, _, _, gy, gx = cf.shape
    coords = zip(vis.vis, vis.weight, pfreq_grid, pu_grid, pu_offset, pv_grid, pv_offset, pw_grid)
    griddata.data[...] = 0.0
    
    dx = gx // 2
    dy = gy // 2
    for v, vwt, chan, xx, xxf, yy, yyf, zz in coords:
        griddata.data[chan, :, zz, (yy - dy):(yy + dy), (xx - dx):(xx + dx)] += \
            cf.data[chan, :, zz, yyf, xxf, :, :] * (numpy.conjugate(v) * vwt)[:, numpy.newaxis, numpy.newaxis]
        sumwt[chan, :] += vwt
    
    im_data = numpy.real(fft(griddata.data))[:, :, 0, ...] * gcf.data
    im = create_image_from_array(im_data, griddata.projection_wcs, griddata.polarisation_frame)
    
    return im, sumwt


def grid_visibility_to_griddata_fast(vis, griddata, cf, gcf):
    """Grid Visibility onto a GridData

    :param vis: Visibility to be gridded
    :param griddata: GridData
    :param kwargs:
    :return: GridData
    """
    nchan, npol, nz, ny, nx = griddata.shape
    sumwt = numpy.zeros([nchan, npol])
    pu_grid, pu_offset, pv_grid, pv_offset, pw_grid, pw_fraction, pfreq_grid = convolution_mapping(vis, griddata, cf)
    coords = zip(vis.vis, vis.weight, pfreq_grid, pu_grid, pv_grid, pw_grid)
    griddata.data[...] = 0.0
    
    for v, vwt, chan, xx, yy, zz in coords:
        griddata.data[chan, :, zz, yy, xx] += numpy.conjugate(v) * vwt
        sumwt[chan, :] += vwt
    
    im_data = numpy.real(fft(griddata.data))[:, :, 0, ...] * gcf.data
    im = create_image_from_array(im_data, griddata.projection_wcs, griddata.polarisation_frame)
    
    return im, sumwt


def degrid_visibility_from_griddata(vis, griddata, cf_griddata, **kwargs):
    """Degrid Visibility from a GridData

    :param vis: Visibility to be degridded
    :param griddata: GridData containing image
    :param cf: Convolution function (as GridData)
    :param kwargs:
    :return: Visibility
    """
    return vis
