"""
Functions that define and manipulate kernels

"""
import logging

import numpy

from data_models.memory_data_models import Image
from libs.fourier_transforms.convolutional_gridding import coordinates, grdsf, w_beam
from libs.fourier_transforms.fft_support import ifft
from libs.image.operations import create_image_from_array, copy_image
from processing_components.convolution_function.operations import create_convolutionfunction_from_image
from processing_components.image.operations import reproject_image

log = logging.getLogger(__name__)


def create_box_convolutionfunction(im, oversampling=1, support=1):
    """ Fill a box car function into a ConvolutionFunction

    Also returns the gridding correction function as an image

    :param im: Image template
    :param oversampling: Oversampling of the convolution function in uv space
    :return: gridding correction Image, gridding kernel as ConvolutionFunction
    """
    assert isinstance(im, Image)
    cf = create_convolutionfunction_from_image(im, oversampling=1, support=1)
    
    nchan, npol, _, _ = im.shape
    
    cf.data[...] = 1.0
    # Now calculate the gridding correction function as an image with the same coordinates as the image
    # which is necessary so that the correction function can be applied directly to the image
    nchan, npol, ny, nx = im.data.shape
    nu = numpy.abs(coordinates(nx))
    
    gcf1d = numpy.sinc(nu)
    gcf = numpy.outer(gcf1d, gcf1d)
    gcf = 1.0 / gcf
    
    gcf_data = numpy.zeros_like(im.data)
    gcf_data[...] = gcf[numpy.newaxis, numpy.newaxis, ...]
    gcf_image = create_image_from_array(gcf_data, cf.projection_wcs, im.polarisation_frame)
    
    return gcf_image, cf


def create_pswf_convolutionfunction(im, oversampling=8, support=6):
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
    
    kernel = numpy.zeros([oversampling, support])
    for grid in range(support):
        for subsample in range(oversampling):
            nu = ((grid - support // 2) - \
                  (subsample - oversampling // 2) / oversampling)
            kernel[subsample, grid] = grdsf([nu / (support // 2)])[1]
    
    kernel /= numpy.sum(numpy.real(kernel[oversampling // 2, :]))
    
    nchan, npol, _, _ = im.shape
    
    cf.data = numpy.zeros([nchan, npol, 1, oversampling, oversampling, support, support]).astype('complex')
    for y in range(oversampling):
        for x in range(oversampling):
            cf.data[:, :, 0, y, x, :, :] = numpy.outer(kernel[y], kernel[x])[numpy.newaxis, numpy.newaxis, ...]
    
    # Now calculate the gridding correction function as an image with the same coordinates as the image
    # which is necessary so that the correction function can be applied directly to the image
    nchan, npol, ny, nx = im.data.shape
    nu = numpy.abs(2.0 * coordinates(nx))
    gcf1d = grdsf(nu)[0]
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
    
    assert isinstance(im, Image)
    # Calculate the convolution kernel. We oversample in u,v space by the factor oversampling
    cf = create_convolutionfunction_from_image(im, oversampling=oversampling, support=support)
    
    cf_shape = list(cf.data.shape)
    cf_shape[2] = nw
    cf.data = numpy.zeros(cf_shape).astype('complex')
    
    cf.grid_wcs.wcs.crpix[4] = nw // 2 + 1.0
    cf.grid_wcs.wcs.cdelt[4] = wstep
    cf.grid_wcs.wcs.ctype[4] = 'WW'
    if numpy.abs(wstep) > 0.0:
        w_list = cf.grid_wcs.sub([5]).wcs_pix2world(range(nw), 0)[0]
    else:
        w_list = [0.0]
    
    assert isinstance(oversampling, int)
    assert oversampling > 0
    
    qnx = nx // oversampling
    qny = ny // oversampling
    
    # Find the actual cellsizes in x and y (radians) after over oversampling (in uv space)
    cell = d2r * im.wcs.wcs.cdelt[1]
    ccell = nx * cell / qnx
    fov = qnx * ccell
    
    cf.data[...] = 0.0
    subim = copy_image(im)
    subim.data = numpy.zeros([nchan, npol, qny, qnx])
    subim.wcs.wcs.crpix[0] -= nx // 2 - qnx // 2
    subim.wcs.wcs.crpix[1] -= ny // 2 - qny // 2
    
    if use_aaf:
        this_pswf_gcf, _ = create_pswf_convolutionfunction(subim, oversampling=1, support=6)
        norm = 1.0 / this_pswf_gcf.data
    else:
        norm = 1.0
    
    if isinstance(pb, Image):
        rpb = reproject_image(pb, subim.wcs, shape=subim.shape)[0]
        norm *= rpb.data
    
    # We might need to work with a larger image
    anx = max(nx, 2 * oversampling * support)
    any = max(ny, 2 * oversampling * support)
    iystart = any // 2 - qny // 2
    iyend = any // 2 + qny // 2
    ixstart = anx // 2 - qnx // 2
    ixend = anx // 2 + qnx // 2
    ycen = any // 2
    xcen = anx // 2
    assert support * oversampling <= anx // 2
    assert support * oversampling <= any // 2
    thisplane = numpy.zeros([nchan, npol, any, anx]).astype('complex')
    for z, w in enumerate(w_list):
        thisplane[..., iystart:iyend, ixstart:ixend] = norm * w_beam(qnx, fov, w)
        thisplane = ifft(thisplane)
        
        for y in range(oversampling):
            vv = range(y + ycen - support * oversampling // 2,
                       y + ycen + support * oversampling // 2, oversampling)[::-1]
            for x in range(oversampling):
                uu = range(x + xcen - support * oversampling // 2,
                           x + xcen + support * oversampling // 2, oversampling)[::-1]
                for chan in range(nchan):
                    for pol in range(npol):
                        cf.data[chan, pol, z, y, x, :, :] = thisplane[chan, pol, :, :][vv, :][:, uu]
        
    pswf_gcf, _ = create_pswf_convolutionfunction(im, oversampling=1, support=6)
    
    return pswf_gcf, cf
