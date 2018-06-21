"""
Functions that define and manipulate kernels

"""
import logging
import copy

import numpy

from data_models.memory_data_models import Image
from libs.fourier_transforms.convolutional_gridding import coordinates, grdsf, w_beam
from libs.image.operations import create_image_from_array
from libs.fourier_transforms.fft_support import ifft, fft

log = logging.getLogger(__name__)

def create_pswf_kernel(gd, oversampling=16, support=3):
    """ Fill an Anti-Aliasing filter into a GridData

    Fill the Prolate Spheroidal Wave Function into a GriData with the specified oversampling.
    Also return the gridding correction function as an image

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
    gend   = nx // 2 + support * oversampling
    
    newgd.grid_wcs.wcs.cdelt[0] /= float(oversampling)
    newgd.grid_wcs.wcs.cdelt[1] /= float(oversampling)
    
    newgd.projection_wcs.wcs.cdelt[0] *= float(oversampling)
    newgd.projection_wcs.wcs.cdelt[1] *= float(oversampling)
    
    newgd.data[...] = 0.0
    newgd.data[:, :, :, gstart:gend, gstart:gend] = kernel[numpy.newaxis, numpy.newaxis, numpy.newaxis, :, :]
    
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


def create_wterm_kernel(gd, oversampling=16, support=3, use_aaf=True):
    """ Fill w projection kernel into a GridData

    :param gd: GridData template
    :param oversampling: Oversampling of the convolution function in uv space
    :return: gridding correction Image, gridding kernel as GridData
    """

    newgd = copy.deepcopy(gd)

    # The convolution correction function is just that for the PSWF
    pswf_gcf, _ = create_pswf_kernel(newgd, oversampling=1, support=support)
    
    nchan, npol, nz, ny, nx = newgd.data.shape

    newgd.grid_wcs.wcs.cdelt[0] /= float(oversampling)
    newgd.grid_wcs.wcs.cdelt[1] /= float(oversampling)
    
    newgd.projection_wcs.wcs.cdelt[0] *= float(oversampling)
    newgd.projection_wcs.wcs.cdelt[1] *= float(oversampling)
    
    w_list = newgd.grid_wcs.sub([3]).wcs_pix2world(range(nz), 0)[0]
    
    newgd.data[...] = 0.0
    if use_aaf:
        norm = 1.0 / pswf_gcf.data
    else:
        norm = 1.0
        
    d2r = numpy.pi / 180.0
    fov = nx * d2r * newgd.grid_wcs.wcs.cdelt[1]


    for z, w in enumerate(w_list):
        newgd.data[:,:,z,:,:] = ifft(w_beam(nx, fov, w) * norm)

    return pswf_gcf, newgd


def create_awterm_kernel(gd, pb=None, oversampling=16, support=3, use_aaf=True):
    """ Fill AW projection kernel into a GridData.

    :param gd: GridData template
    :param pb: Primary beam model image
    :param oversampling: Oversampling of the convolution function in uv space
    :return: gridding correction Image, gridding kernel as GridData
    """
    
    pswf_gcf, _ = create_pswf_kernel(gd, oversampling=1, support=support)
    
    nchan, npol, nz, ny, nx = gd.data.shape
    
    newgd =  copy.deepcopy(gd)
    
    newgd.grid_wcs.wcs.cdelt[0] /= float(oversampling)
    newgd.grid_wcs.wcs.cdelt[1] /= float(oversampling)
    
    newgd.projection_wcs.wcs.cdelt[0] *= float(oversampling)
    newgd.projection_wcs.wcs.cdelt[1] *= float(oversampling)
    
    w_list = newgd.grid_wcs.sub([3]).wcs_pix2world(range(nz), 0)[0]

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
