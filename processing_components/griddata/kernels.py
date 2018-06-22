"""
Functions that define and manipulate kernels

"""
import copy
import logging

import numpy

from data_models.memory_data_models import Image
from libs.fourier_transforms.convolutional_gridding import coordinates, grdsf, w_beam
from libs.fourier_transforms.fft_support import ifft
from libs.image.operations import create_image_from_array

log = logging.getLogger(__name__)


def create_pswf_kernel(gd, oversampling=8, support=6):
    """ Fill an Anti-Aliasing filter into a GridData

    Fill the Prolate Spheroidal Wave Function into a GriData with the specified oversampling. Only the inner
    non-zero part is retained
    
    Also returns the gridding correction function as an image

    :param gd: GridData template
    :param oversampling: Oversampling of the convolution function in uv space
    :return: gridding correction Image, gridding kernel as GridData
    """
    
    # Calculate the convolution kernel. We oversample in u,v space by the factor oversampling
    newgd = copy.deepcopy(gd)
    nchan, npol, nz, ny, nx = newgd.data.shape
    nu = numpy.arange(-support, +support, 1.0 / oversampling)
    _, kernel1d = grdsf(nu / support)
    kernel = numpy.outer(kernel1d, kernel1d)
    kernel[kernel > 0.0] = kernel[kernel > 0.0] / kernel.max()
    
    gstart = nx // 2 - support * oversampling
    gend = nx // 2 + support * oversampling
    
    newgd.grid_wcs.wcs.cdelt[0] /= float(oversampling)
    newgd.grid_wcs.wcs.cdelt[1] /= float(oversampling)
    
    newgd.projection_wcs.wcs.cdelt[0] *= float(oversampling)
    newgd.projection_wcs.wcs.cdelt[1] *= float(oversampling)
    newgd.projection_wcs.wcs.crpix[0] -= gstart
    newgd.projection_wcs.wcs.crpix[1] -= gstart
    
    newgd.data = kernel[numpy.newaxis, numpy.newaxis, numpy.newaxis, :, :]
    
    # Now calculate the gridding correction function as an image with the same coordinates as gd.projection_wcs
    # which is necessary so that the correction function can be applied directly to the image
    nu = numpy.abs(2.0 * coordinates(nx))
    gcf1d, _ = grdsf(nu)
    gcf = numpy.outer(gcf1d, gcf1d)
    gcf[gcf > 0.0] = gcf.max() / gcf[gcf > 0.0]
    
    gcf_data = numpy.zeros([nchan, npol, ny, nx])
    gcf_data[...] = gcf[numpy.newaxis, numpy.newaxis, ...]
    gcf_image = create_image_from_array(gcf_data, gd.projection_wcs, gd.polarisation_frame)
    
    return gcf_image, newgd


def create_wterm_kernel(gd, nw=1, wstep=0.0, oversampling=8, support=6, use_aaf=True):
    """ Fill w projection kernel into a GridData

    :param gd: GridData template
    :param oversampling: Oversampling of the convolution function in uv space
    :return: gridding correction Image, gridding kernel as GridData
    """
    newgd = copy.deepcopy(gd)
    
    nchan, npol, _, ny, nx = newgd.shape
    newgd.data = numpy.zeros([nchan, npol, nw, ny, nx], dtype='complex')
    
    newgd.grid_wcs.wcs.crpix[2] = nw / 2.0
    newgd.grid_wcs.wcs.cdelt[2] = wstep
    
    # The convolution correction function is just that for the PSWF
    pswf_gcf, _ = create_pswf_kernel(newgd, oversampling=1, support=support)
    
    newgd.grid_wcs.wcs.cdelt[0] /= float(oversampling)
    newgd.grid_wcs.wcs.cdelt[1] /= float(oversampling)
    
    newgd.projection_wcs.wcs.cdelt[0] *= float(oversampling)
    newgd.projection_wcs.wcs.cdelt[1] *= float(oversampling)
    
    w_list = newgd.grid_wcs.sub([3]).wcs_pix2world(range(nw), 0)[0]
    
    newgd.data[...] = 0.0
    if use_aaf:
        norm = 1.0 / pswf_gcf.data
    else:
        norm = 1.0
    
    d2r = numpy.pi / 180.0
    fov = nx * d2r * newgd.grid_wcs.wcs.cdelt[1]
    
    for z, w in enumerate(w_list):
        newgd.data[:, :, z, :, :] = ifft(w_beam(nx, fov, w) * norm)
    
    return pswf_gcf, newgd


def create_awterm_kernel(gd, pb=None, nw=1, wstep=0.0, oversampling=8, support=6, use_aaf=True):
    """ Fill AW projection kernel into a GridData.

    :param gd: GridData template
    :param pb: Primary beam model image
    :param oversampling: Oversampling of the convolution function in uv space
    :return: gridding correction Image, gridding kernel as GridData
    """
    
    pswf_gcf, _ = create_pswf_kernel(gd, oversampling=1, support=support)
    
    newgd = copy.deepcopy(gd)
    
    nchan, npol, _, ny, nx = gd.data.shape
    
    newgd.grid_wcs.wcs.cdelt[0] /= float(oversampling)
    newgd.grid_wcs.wcs.cdelt[1] /= float(oversampling)
    
    newgd.projection_wcs.wcs.cdelt[0] *= float(oversampling)
    newgd.projection_wcs.wcs.cdelt[1] *= float(oversampling)
    
    nchan, npol, _, ny, nx = newgd.shape
    newgd.data = numpy.zeros([nchan, npol, nw, ny, nx], dtype='complex')
    
    newgd.grid_wcs.wcs.crpix[2] = nw / 2.0
    newgd.grid_wcs.wcs.cdelt[2] = wstep
    
    w_list = newgd.grid_wcs.sub([3]).wcs_pix2world(range(nw), 0)[0]
    
    d2r = numpy.pi / 180.0
    fov = nx * d2r * newgd.grid_wcs.wcs.cdelt[1]
    
    newgd.data[...] = 0.0
    if use_aaf:
        norm = 1.0 / pswf_gcf.data
    else:
        norm = 1.0
    if isinstance(pb, Image):
        norm *= pb.data
    
    for z, w in enumerate(w_list):
        newgd.data[:, :, z, :, :] = ifft(w_beam(nx, fov, w) * norm)
    
    return pswf_gcf, newgd