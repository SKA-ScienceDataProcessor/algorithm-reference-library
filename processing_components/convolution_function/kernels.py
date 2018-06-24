"""
Functions that define and manipulate kernels

"""
import copy
import logging

import numpy

from data_models.memory_data_models import Image
from libs.fourier_transforms.convolutional_gridding import coordinates, grdsf, w_beam
from processing_components.convolution_function.operations import create_convolutionfunction_from_image
from libs.fourier_transforms.fft_support import ifft
from libs.image.operations import create_image_from_array, copy_image

log = logging.getLogger(__name__)

def create_pswf_convolutionfunction(im, oversampling=8, support=3):
    """ Fill an Anti-Aliasing filter into a ConvolutionFunction

    Fill the Prolate Spheroidal Wave Function into a GriData with the specified oversampling. Only the inner
    non-zero part is retained
    
    Also returns the gridding correction function as an image

    :param im: Image template
    :param oversampling: Oversampling of the convolution function in uv space
    :return: gridding correction Image, gridding kernel as ConvolutionFunction
    """
    assert isinstance(im, Image)
    # Calculate the convolution kernel. We oversample in u,v space by the factor oversampling
    cf = create_convolutionfunction_from_image(im, oversampling=oversampling, support=support)
    
    nu = numpy.linspace(-support, support, oversampling*2*support + 1)
    _, kernel1d = grdsf(nu / support)
    kernel1d /= kernel1d.max()
    
    nchan, npol, _, _ = im.shape
    
    cf.data = numpy.zeros([nchan, npol, 1, oversampling, oversampling, 2 * support, 2 * support])
    for y in range(oversampling):
        vv = range(y, y+2*support*oversampling, oversampling)[::-1]
        for x in range(oversampling):
            uu = range(x, x+2*support*oversampling, oversampling)[::-1]
            cf.data[:, :, 0, y, x, :, :] = numpy.outer(kernel1d[vv], kernel1d[uu])[numpy.newaxis, numpy.newaxis, ...]
    
    # Now calculate the gridding correction function as an image with the same coordinates as the image
    # which is necessary so that the correction function can be applied directly to the image
    nchan, npol, ny, nx = im.data.shape
    nu = numpy.abs(2.0 * coordinates(nx))
    gcf1d, _ = grdsf(nu)
    gcf = numpy.outer(gcf1d, gcf1d)
    gcf[gcf > 0.0] = gcf.max() / gcf[gcf > 0.0]
    
    gcf_data = numpy.zeros_like(im.data)
    gcf_data[...] = gcf[numpy.newaxis, numpy.newaxis, ...]
    gcf_image = create_image_from_array(gcf_data, cf.projection_wcs, im.polarisation_frame)
    
    return gcf_image, cf

def create_awterm_convolutionfunction(im, pb=None, nw=1, wstep=0.0, oversampling=8, support=6, use_aaf=True):
    """ Fill AW projection kernel into a GridData.

    :param im: Image template
    :param pb: Primary beam model image
    :param nw: Number of w planes
    :param wstep: Step in w (wavelengths)
    :param oversampling: Oversampling of the convolution function in uv space
    :return: gridding correction Image, gridding kernel as GridData
    """
    d2r = numpy.pi / 180.0

    # We only need the gridding correction function for the PSWF so we make
    # it for the shape of the image
    nchan, npol, ny, nx = im.data.shape

    pswf_gcf, cf = create_pswf_convolutionfunction(im, oversampling=1, support=6)

    cf_shape = list(cf.data.shape)
    cf_shape[2] = nw
    cf.data = numpy.zeros(cf_shape)

    cf.grid_wcs.wcs.crpix[4] = nw // 2
    cf.grid_wcs.wcs.cdelt[4] = wstep
    cf.grid_wcs.wcs.ctype[4] = 'WW'
    w_list = cf.grid_wcs.sub([5]).wcs_pix2world(range(nw), 1)[0]

    qnx = nx // oversampling
    qny = ny // oversampling

    # Find the actual cellsizes in x and y (radians) after over oversampling (in uv space)
    cell = d2r * im.wcs.wcs.cdelt[1]
    ccell = nx * cell / qnx
    fov = qnx * ccell

    iystart = ny // 2 - qny// 2
    iyend = ny // 2 + qny // 2
    ixstart = nx // 2 - qnx // 2
    ixend = nx // 2 + qnx // 2


    cf.data[...] = 0.0
    if use_aaf:
        subim = copy_image(im)
        subim.data = numpy.zeros([nchan, npol, qny, qnx])
        this_pswf_gcf, cf = create_pswf_convolutionfunction(subim, oversampling=1, support=6)
        norm = 1.0 / this_pswf_gcf.data
    else:
        norm = 1.0
        
    if isinstance(pb, Image):
        norm *= pb.data

    cf.data = numpy.zeros([nchan, npol, nw, oversampling, oversampling, 2 * support, 2 * support]).astype('complex')
    
    thisplane = numpy.zeros([nchan, npol, ny, nx]).astype('complex')
    for z, w in enumerate(w_list):
        thisplane[..., iystart:iyend, ixstart:ixend] = norm * w_beam(qnx, fov, w)
        thisplane = ifft(thisplane)

        for y in range(oversampling):
            vv = slice(y, y + 2 * support * oversampling, oversampling)
            for x in range(oversampling):
                uu = slice(x, x + 2 * support * oversampling, oversampling)
                cf.data[:, :, z, y, x, :, :] = thisplane[:, :, vv, uu]

    return pswf_gcf, cf