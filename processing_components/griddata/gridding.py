""" Imaging is based on used of the FFT to perform Fourier transforms efficiently. Since the observed visibility data_models
do not arrive naturally on grid points, the sampled points are resampled on the FFT grid using a convolution function to
smear out the sample points. The resulting grid points are then FFT'ed. The result can be corrected for the griddata
convolution function by division in the image plane of the transform.

This module contains functions for performing the griddata process and the inverse degridding process.

The GridData data model is used to hold the specification of the desired result.
"""

import logging

import numpy
import numpy.testing

from processing_components.visibility.operations import copy_visibility
from processing_library.image.operations import ifft, fft, create_image_from_array
from processing_components.griddata.operations import copy_griddata

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
    # We use the grid_wcs's to do the coordinate conversion
    # Find the nearest grid points
    pu_grid, pv_grid = \
        numpy.round(griddata.grid_wcs.sub([1, 2]).wcs_world2pix(vis.uvw[:, 0], vis.uvw[:, 1], 0)).astype('int')
    assert numpy.min(pu_grid) >= 0
    assert numpy.max(pu_grid) < griddata.shape[3], "U axis overflows: %f" % numpy.max(pu_grid)
    assert numpy.min(pv_grid) >= 0
    assert numpy.max(pv_grid) < griddata.shape[4], "V axis overflows: %f" % numpy.max(pv_grid)
    
    # We now have the location of grid points, convert back to uv space and find the remainder (in wavelengths). We
    # then use this to calculate the subsampling indices (DUU, DVV)
    wu_grid, wv_grid = griddata.grid_wcs.sub([1, 2]).wcs_pix2world(pu_grid, pv_grid, 0)
    wu_subsample, wv_subsample = vis.uvw[:, 0] - wu_grid, vis.uvw[:, 1] - wv_grid
    
    pu_offset, pv_offset = \
        numpy.floor(cf.grid_wcs.sub([3, 4]).wcs_world2pix(wu_subsample, wv_subsample, 0)).astype('int')
    
    ###### W mapping for Grid
    # nchan, npol, w, v, u
    pwg_pixel = griddata.grid_wcs.sub([3]).wcs_world2pix(vis.uvw[:, 2], 0)[0]
    # Find the nearest grid point
    pwg_grid = numpy.round(pwg_pixel).astype('int')
    assert numpy.min(pwg_grid) >= 0
    assert numpy.max(pwg_grid) < cf.shape[2], "W axis overflows: %f" % numpy.max(pwg_grid)
    pwg_fraction = pwg_pixel - pwg_grid
    
    ###### W mapping for CF
    # nchan, npol, w, dv, du, v, u
    pwc_pixel = cf.grid_wcs.sub([5]).wcs_world2pix(vis.uvw[:, 2], 0)[0]
    pwc_grid = numpy.round(pwc_pixel).astype('int')
    assert numpy.min(pwc_grid) >= 0, "W axis overflows: %f" % numpy.max(pwc_grid)
    assert numpy.max(pwc_grid) < cf.shape[2], "W axis overflows: %f" % numpy.max(pwc_grid)
    pwc_fraction = pwc_pixel - pwc_grid
    
    ###### Frequency mapping
    pfreq_pixel = griddata.grid_wcs.sub([5]).wcs_world2pix(vis.frequency, 0)[0]
    # Find the nearest grid point
    pfreq_grid = numpy.round(pfreq_pixel).astype('int')
    pfreq_fraction = pfreq_pixel - pfreq_grid
    if numpy.max(numpy.abs(pfreq_fraction)) > channel_tolerance:
        log.warning("convolution_mapping: alignment of visibility and image grids exceeds tolerance %s" %
                    (numpy.max(pfreq_fraction)))
    
    ######  TODO: Polarisation mapping
    
    return pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction, pfreq_grid


def grid_visibility_to_griddata(vis, griddata, cf):
    """Grid Visibility onto a GridData

    :param vis: Visibility to be gridded
    :param griddata: GridData
    :param cf: Convolution function
    :param kwargs:
    :return: GridData
    """
    nchan, npol, nz, oversampling, _, support, _ = cf.shape
    sumwt = numpy.zeros([nchan, npol])
    pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction, pfreq_grid = \
        convolution_mapping(vis, griddata, cf)
    _, _, _, _, _, gv, gu = cf.shape
    coords = zip(vis.vis * vis.weight, vis.weight, pfreq_grid, pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid,
                 pwc_grid)
    griddata.data[...] = 0.0
    
    # Do this in place to avoid creating a new copy. Doing the conjugation outside the loop
    # reduces run time immensely
    cf.data = numpy.conjugate(cf.data)
    
    du = gu // 2
    dv = gv // 2
    for v, vwt, chan, uu, uuf, vv, vvf, zzg, zzc in coords:
        griddata.data[chan, :, zzg, (vv - dv):(vv + dv), (uu - du):(uu + du)] += \
            cf.data[chan, :, zzc, vvf, uuf, :, :] * v[:, numpy.newaxis, numpy.newaxis]
        sumwt[chan, :] += vwt
    
    cf.data = numpy.conjugate(cf.data)
    return griddata, sumwt


def grid_visibility_to_griddata_fast(vis, griddata, cf, gcf):
    """Grid Visibility onto a GridData

    :param vis: Visibility to be gridded
    :param griddata: GridData
    :param kwargs:
    :return: GridData
    """
    nchan, npol, nz, ny, nx = griddata.shape
    sumwt = numpy.zeros([nchan, npol])
    pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction, pfreq_grid = \
        convolution_mapping(vis, griddata, cf)
    _, _, _, _, _, gv, gu = cf.shape
    coords = zip(vis.vis, vis.weight, pfreq_grid, pu_grid, pv_grid, pwg_grid)
    griddata.data[...] = 0.0
    
    for v, vwt, chan, xx, yy, zzg in coords:
        griddata.data[chan, :, zzg, yy, xx] += v * vwt
        sumwt[chan, :] += vwt
    
    projected = numpy.sum(griddata.data, axis=2)
    im_data = numpy.real(fft(projected)) * gcf.data
    im = create_image_from_array(im_data, griddata.projection_wcs, griddata.polarisation_frame)
    
    return im, sumwt


def grid_weight_to_griddata(vis, griddata, cf):
    """Grid Visibility weight onto a GridData

    :param vis: Visibility to be gridded
    :param griddata: GridData
    :param kwargs:
    :return: GridData
    """
    nchan, npol, nz, ny, nx = griddata.shape
    sumwt = numpy.zeros([nchan, npol])
    pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction, pfreq_grid = \
        convolution_mapping(vis, griddata, cf)
    _, _, _, _, _, gv, gu = cf.shape
    coords = zip(vis.weight, pfreq_grid, pu_grid, pv_grid, pwg_grid)
    griddata.data[...] = 0.0
    
    for vwt, chan, xx, yy, zzg in coords:
        griddata.data[chan, :, zzg, yy, xx] += vwt
        sumwt[chan, :] += vwt
    
    return griddata, sumwt

def griddata_merge_weights(gd_list, algorithm='uniform'):
    """ Merge weights into one grid
    
    :param gd_list:
    :param gd:
    :param algorithm:
    :return:
    """
    centre = len(gd_list) // 2
    gd = copy_griddata(gd_list[centre][0])
    sumwt = gd_list[centre][1]
    
    frequency = 0.0
    bandwidth = 0.0
    
    for i, g in enumerate(gd_list):
        if i!=centre:
            gd.data += g[0].data
            sumwt += g[1]
        frequency += g[0].grid_wcs.wcs.crval[4]
        bandwidth += g[0].grid_wcs.wcs.cdelt[4]
    
    gd.grid_wcs.wcs.cdelt[4] = bandwidth
    gd.grid_wcs.wcs.crval[4] = frequency / len(gd_list)
    return gd, sumwt

def griddata_reweight(vis, griddata, cf):
    """Reweight Grid Visibility weight using the weights in griddata

    :param vis: Visibility to be reweighted
    :param griddata: GridData, sumwt
    :param kwargs:
    :return: GridData
    """
    nchan, npol, nz, ny, nx = griddata.shape
    pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction, pfreq_grid = \
        convolution_mapping(vis, griddata, cf)
    _, _, _, _, _, gv, gu = cf.shape
    coords = zip(vis.weight, pfreq_grid, pu_grid, pv_grid, pwg_grid)
    
    for vwt, chan, xx, yy, zzg in coords:
        if numpy.real(griddata.data[chan, :, zzg, yy, xx]).all() > 0.0:
            vwt /= numpy.real(griddata.data[chan, :, zzg, yy, xx])
    
    return vis

def degrid_visibility_from_griddata(vis, griddata, cf, **kwargs):
    """Degrid Visibility from a GridData

    :param vis: Visibility to be degridded
    :param griddata: GridData containing image
    :param cf: Convolution function (as GridData)
    :param kwargs:
    :return: Visibility
    """
    nchan, npol, nz, oversampling, _, support, _ = cf.shape
    pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction, pfreq_grid = \
        convolution_mapping(vis, griddata, cf)
    _, _, _, _, _, gv, gu = cf.shape
    _, _, _, _, _, gv, gu = cf.shape
    
    newvis = copy_visibility(vis, zero=True)
    
    # coords = zip(pfreq_grid, pu_grid, pu_offset, pv_grid, pv_offset, pw_grid)
    
    du = gu // 2
    dv = gv // 2
    
    nvis = vis.vis.shape[0]
    
    # TODO: Optimise
    for i in range(nvis):
        chan, uu, uuf, vv, vvf, zzg, zzc = pfreq_grid[i], pu_grid[i], pu_offset[i], pv_grid[i], pv_offset[i], \
                                           pwg_grid[i], pwc_grid[i]
        # Use einsum to replace the following:
        # newvis.vis[i,:] = numpy.sum(griddata.data[chan, :, zzg, (vv - dv):(vv + dv), (uu - du):(uu + du)] *
        #                              cf.data[chan, :, zzc, vvf, uuf, :, :], axis=(1, 2))
    
        newvis.vis[i, :] = numpy.einsum('ijk,ijk->i',
                                        griddata.data[chan, :, zzg, (vv - dv):(vv + dv), (uu - du):(uu + du)],
                                        cf.data[chan, :, zzc, vvf, uuf, :, :])
        
    return newvis


def fft_griddata_to_image(griddata, gcf, imaginary=False):
    """

    :param griddata:
    :param gcf: Grid correction image
    :return:
    """
    
    projected = numpy.sum(griddata.data, axis=2)
    _, _, ny, nx = projected.data.shape
    
    im_data = ifft(projected) * gcf.data * float(nx) * float(ny)
    
    im_real = create_image_from_array(im_data.real, griddata.projection_wcs, griddata.polarisation_frame)
    
    if imaginary:
        im_imag = create_image_from_array(im_data.imag, griddata.projection_wcs, griddata.polarisation_frame)
        return im_real, im_imag
    else:
        return im_real


def fft_image_to_griddata(im, griddata, gcf):
    """Fill griddata with transform of im

    :param griddata:
    :param gcf: Grid correction image
    :return:
    """
    # chan, pol, z, u, v, w
    griddata.data[:, :, :, ...] = fft(im.data * gcf.data)[:, :, numpy.newaxis, ...]
    
    return griddata
