"""
Functions that define and manipulate kernels

"""

__all__ = ['create_pswf_convolutionfunction', 'create_box_convolutionfunction', 'create_awterm_convolutionfunction']

import logging

import numpy

from arl.data_models.memory_data_models import Image
from arl.processing_library.fourier_transforms.convolutional_gridding import coordinates, grdsf
from arl.processing_library.image.operations import copy_image, create_w_term_like, pad_image, fft_image
from arl.processing_library.image.operations import create_image_from_array
from arl.processing_components.griddata.convolution_functions import create_convolutionfunction_from_image
from arl.processing_components.image.operations import reproject_image, create_empty_image_like

log = logging.getLogger(__name__)


def create_box_convolutionfunction(im, oversampling=1, support=1):
    """ Fill a box car function into a ConvolutionFunction

    Also returns the griddata correction function as an image

    :param im: Image template
    :param oversampling: Oversampling of the convolution function in uv space
    :return: griddata correction Image, griddata kernel as ConvolutionFunction
    """
    assert isinstance(im, Image)
    cf = create_convolutionfunction_from_image(im, oversampling=1, support=4)
    
    nchan, npol, _, _ = im.shape
    
    cf.data[...] = 0.0 + 0.0j
    cf.data[..., 2, 2] = 1.0 + 0.0j
    
    # Now calculate the griddata correction function as an image with the same coordinates as the image
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

    Also returns the griddata correction function as an image

    :param im: Image template
    :param oversampling: Oversampling of the convolution function in uv space
    :return: griddata correction Image, griddata kernel as ConvolutionFunction
    """
    assert isinstance(im, Image), im
    # Calculate the convolution kernel. We oversample in u,v space by the factor oversampling
    cf = create_convolutionfunction_from_image(im, oversampling=oversampling, support=support)
    
    kernel = numpy.zeros([oversampling, support])
    for grid in range(support):
        for subsample in range(oversampling):
            nu = ((grid - support // 2) - (subsample - oversampling // 2) / oversampling)
            kernel[subsample, grid] = grdsf([nu / (support // 2)])[1]
    
    kernel /= numpy.sum(numpy.real(kernel[oversampling // 2, :]))
    
    nchan, npol, _, _ = im.shape
    
    cf.data = numpy.zeros([nchan, npol, 1, oversampling, oversampling, support, support]).astype('complex')
    for y in range(oversampling):
        for x in range(oversampling):
            cf.data[:, :, 0, y, x, :, :] = numpy.outer(kernel[y, :], kernel[x, :])[numpy.newaxis, numpy.newaxis, ...]
    norm = numpy.sum(numpy.real(cf.data[0, 0, 0, 0, 0, :, :]))
    cf.data /= norm
    
    # Now calculate the griddata correction function as an image with the same coordinates as the image
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


def create_awterm_convolutionfunction(im, make_pb=None, nw=1, wstep=1e15, oversampling=8, support=6, use_aaf=True,
                                      maxsupport=512):
    """ Fill AW projection kernel into a GridData.

    :param im: Image template
    :param make_pb: Function to make the primary beam model image (hint: use a partial)
    :param nw: Number of w planes
    :param wstep: Step in w (wavelengths)
    :param oversampling: Oversampling of the convolution function in uv space
    :return: griddata correction Image, griddata kernel as GridData
    """
    d2r = numpy.pi / 180.0
    
    # We only need the griddata correction function for the PSWF so we make
    # it for the shape of the image
    nchan, npol, ony, onx = im.data.shape
    
    assert isinstance(im, Image)
    # Calculate the template convolution kernel.
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

    nx = max(maxsupport, 2 * oversampling * support)
    ny = max(maxsupport, 2 * oversampling * support)
    
    qnx = nx // oversampling
    qny = ny // oversampling

    cf.data[...] = 0.0

    subim = copy_image(im)
    ccell = onx * numpy.abs(d2r * subim.wcs.wcs.cdelt[0]) / qnx

    subim.data = numpy.zeros([nchan, npol, qny, qnx])
    subim.wcs.wcs.cdelt[0] = -ccell / d2r
    subim.wcs.wcs.cdelt[1] = +ccell / d2r
    subim.wcs.wcs.crpix[0] = qnx // 2 + 1.0
    subim.wcs.wcs.crpix[1] = qny // 2 + 1.0

    if use_aaf:
        this_pswf_gcf, _ = create_pswf_convolutionfunction(subim, oversampling=1, support=6)
        norm = 1.0 / this_pswf_gcf.data
    else:
        norm = 1.0
    
    if make_pb is not None:
        pb = make_pb(subim)
        rpb, footprint = reproject_image(pb, subim.wcs, shape=subim.shape)
        rpb.data[footprint.data < 1e-6] = 0.0
        norm *= rpb.data
    
    # We might need to work with a larger image
    padded_shape = [nchan, npol, ny, nx]
    thisplane = copy_image(subim)
    thisplane.data = numpy.zeros(thisplane.shape, dtype='complex')
    for z, w in enumerate(w_list):
        thisplane.data[...] = 0.0 + 0.0j
        thisplane = create_w_term_like(thisplane, w, dopol=True)
        thisplane.data *= norm
        paddedplane = pad_image(thisplane, padded_shape)
        paddedplane = fft_image(paddedplane)
        
        ycen, xcen = ny // 2, nx // 2
        for y in range(oversampling):
            ybeg = y + ycen + (support * oversampling) // 2 - oversampling // 2
            yend = y + ycen - (support * oversampling) // 2 - oversampling // 2
            # vv = range(ybeg, yend, -oversampling)
            for x in range(oversampling):
                xbeg = x + xcen + (support * oversampling) // 2 - oversampling // 2
                xend = x + xcen - (support * oversampling) // 2 - oversampling // 2

                # uu = range(xbeg, xend, -oversampling)
                cf.data[..., z, y, x, :, :] = paddedplane.data[..., ybeg:yend:-oversampling, xbeg:xend:-oversampling]
                # for chan in range(nchan):
                #     for pol in range(npol):
                #         cf.data[chan, pol, z, y, x, :, :] = paddedplane.data[chan, pol, :, :][vv, :][:, uu]

    cf.data /= numpy.sum(numpy.real(cf.data[0, 0, nw // 2, oversampling // 2, oversampling // 2, :, :]))
    cf.data = numpy.conjugate(cf.data)
    
    if use_aaf:
        pswf_gcf, _ = create_pswf_convolutionfunction(im, oversampling=1, support=6)
    else:
        pswf_gcf = create_empty_image_like(im)
        pswf_gcf.data[...] = 1.0
    
    return pswf_gcf, cf
