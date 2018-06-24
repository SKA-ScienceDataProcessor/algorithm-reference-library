#
"""
Functions that define and manipulate ConvolutionFunctions.

The griddata has axes [chan, pol, z, dy, dx, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
order in the WCS is reversed so the grid_WCS describes UU, VV, DUU, DVV, WW, STOKES, FREQ axes.

GridData can be used to hold the Fourier transform of an Image or gridded visibilities. In addition, the convolution
function can be stored in a GridData, most probably with finer spatial sampling.


"""
import copy
import logging

import numpy
from astropy.wcs import WCS

from data_models.memory_data_models import GridData, ConvolutionFunction
from data_models.polarisation import PolarisationFrame
from libs.fourier_transforms.fft_support import ifft, fft
from libs.image.operations import create_image_from_array

log = logging.getLogger(__name__)


def griddata_sizeof(gd: GridData):
    """ Return size in GB
    """
    return gd.size()


def create_convolutionfunction_from_array(data: numpy.array, grid_wcs: WCS, projection_wcs: WCS,
                               polarisation_frame: PolarisationFrame) -> ConvolutionFunction:
    """ Create a convolution function from an array and wcs's
    
    The griddata has axes [chan, pol, z, dy, dx, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
    order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes
    
    The axes UU,VV have the same physical stride as the image, The axes DUU, DVV are subsampled.
    
    Convolution function holds the original sky plane projection in the projection_wcs.

    :param data: Numpy.array
    :param grid_wcs: Grid world coordinate system
    :param projection_wcs: Projection world coordinate system
    :param polarisation_frame: Polarisation Frame
    :return: GridData
    
    """
    fconvfunc = ConvolutionFunction()
    fconvfunc.polarisation_frame = polarisation_frame
    
    fconvfunc.data = data
    fconvfunc.grid_wcs = grid_wcs.deepcopy()
    fconvfunc.projection_wcs = projection_wcs.deepcopy()
    
    assert isinstance(fconvfunc, ConvolutionFunction), "Type is %s" % type(fconvfunc)
    return fconvfunc


def create_convolutionfunction_from_image(im: numpy.array, nz=1, zstep=1e-7, ztype='WW', oversampling=8, support=16):
    """ Create a convolution function from an image

    The griddata has axes [chan, pol, z, dy, dx, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
    order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes

    The axes UU,VV have the same physical stride as the image, The axes DUU, DVV are subsampled.

    Convolution function holds the original sky plane projection in the projection_wcs.

    :param im: Image
    :param nz: Number of z axes, usually z is W
    :param zstep: Step in z, usually z is W
    :param ztype: Type of Z, usually 'WW'
    :param oversampling:
    :param support:
    :return: Convolution Function

    """
    assert len(im.shape) == 4
    assert im.wcs.wcs.ctype[0] == 'RA---SIN'
    assert im.wcs.wcs.ctype[1] == 'DEC--SIN'

    d2r = numpy.pi / 180.0

    # WCS Coords are [x, y, dy, dx, z, pol, chan] where x, y, z are spatial axes in real space or Fourier space
    # Array Coords are [chan, pol, z, dy, dx, y, x] where x, y, z are spatial axes in real space or Fourier space
    cf_wcs = WCS(naxis=7)

    cf_wcs.wcs.ctype[0] = 'UU'
    cf_wcs.wcs.ctype[1] = 'VV'
    cf_wcs.wcs.ctype[2] = 'DUU'
    cf_wcs.wcs.ctype[3] = 'DVV'
    cf_wcs.wcs.ctype[4] = ztype
    cf_wcs.wcs.ctype[5] = im.wcs.wcs.ctype[2]
    cf_wcs.wcs.ctype[6] = im.wcs.wcs.ctype[3]

    cf_wcs.wcs.axis_types[0] = 0
    cf_wcs.wcs.axis_types[1] = 0
    cf_wcs.wcs.axis_types[2] = 0
    cf_wcs.wcs.axis_types[3] = 0
    cf_wcs.wcs.axis_types[4] = 0
    cf_wcs.wcs.axis_types[5] = im.wcs.wcs.axis_types[2]
    cf_wcs.wcs.axis_types[6] = im.wcs.wcs.axis_types[3]

    cf_wcs.wcs.crval[0] = 0.0
    cf_wcs.wcs.crval[1] = 0.0
    cf_wcs.wcs.crval[2] = 0.0
    cf_wcs.wcs.crval[3] = 0.0
    cf_wcs.wcs.crval[4] = 0.0
    cf_wcs.wcs.crval[5] = im.wcs.wcs.crval[2]
    cf_wcs.wcs.crval[6] = im.wcs.wcs.crval[3]

    cf_wcs.wcs.crpix[0] = support / 2.0 + 1.0
    cf_wcs.wcs.crpix[1] = support / 2.0 + 1.0
    cf_wcs.wcs.crpix[2] = oversampling / 2.0 + 1.0
    cf_wcs.wcs.crpix[3] = oversampling / 2.0 + 1.0
    cf_wcs.wcs.crpix[4] = nz // 2 + 1.0
    cf_wcs.wcs.crpix[5] = im.wcs.wcs.crpix[2]
    cf_wcs.wcs.crpix[6] = im.wcs.wcs.crpix[3]

    # The sampling on the UU and VV axes should be the same as for the image.
    # The sampling on the DUU and DVV axes should be oversampling times finer.
    cf_wcs.wcs.cdelt[0] = 1.0 / (im.shape[3] * d2r * im.wcs.wcs.cdelt[0])
    cf_wcs.wcs.cdelt[1] = 1.0 / (im.shape[2] * d2r * im.wcs.wcs.cdelt[1])
    cf_wcs.wcs.cdelt[2] = cf_wcs.wcs.cdelt[0] / oversampling
    cf_wcs.wcs.cdelt[3] = cf_wcs.wcs.cdelt[1] / oversampling
    cf_wcs.wcs.cdelt[4] = zstep
    cf_wcs.wcs.cdelt[5] = im.wcs.wcs.cdelt[2]
    cf_wcs.wcs.cdelt[6] = im.wcs.wcs.cdelt[3]

    grid_data = im.data[..., numpy.newaxis, :, :].astype('complex')
    grid_data[...] = 0.0

    nchan, npol, ny, nx = im.shape
    
    fconvfunc = ConvolutionFunction()
    fconvfunc.polarisation_frame = im.polarisation_frame

    fconvfunc.data = numpy.zeros([nchan, npol, nz, oversampling, oversampling, support, support])
    fconvfunc.grid_wcs = cf_wcs.deepcopy()
    fconvfunc.projection_wcs = im.wcs.deepcopy()
    
    assert isinstance(fconvfunc, ConvolutionFunction), "Type is %s" % type(fconvfunc)
    return fconvfunc


def convert_convolutionfunction_to_image(gd):
    """ Convert griddata to an image
    
    :param gd:
    :return:
    """
    return create_image_from_array(gd.data, gd.grid_wcs, gd.polarisation_frame)
