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

def convert_convolutionfunction_to_image(gd):
    """ Convert griddata to an image
    
    :param gd:
    :return:
    """
    return create_image_from_array(gd.data, gd.grid_wcs, gd.polarisation_frame)


def create_convolutionfunction_from_griddata(griddata, support=3, oversampling=8):
    """Convert a griddate to a convolution function

    :param griddata:
    :return:
    """
    newwcs = WCS(naxis=7)
    
    nchan, npol, nz, ny, nx = griddata.data.shape
    
    newgrid = numpy.zeros([nchan, npol, nz, oversampling, oversampling, support, support], dtype='complex')

    ystart = ny // 2 - oversampling * support // 2
    xstart = nx // 2 - oversampling * support // 2
    yend = ny // 2 + oversampling * support // 2
    xend = nx // 2 + oversampling * support // 2
    for chan in range(nchan):
        for pol in range(npol):
            for z in range(nz):
                for y in range(oversampling):
                    slicey = slice(yend + y, ystart + y, -oversampling)
                    for x in range(oversampling):
                        slicex = slice(xend + x, xstart + x, -oversampling)
                        newgrid[chan, pol, z, y, x, ...] = griddata.data[chan, pol, z, slicey, slicex]


    for axis in range(5):
        newwcs.wcs.ctype[axis + 2] = griddata.grid_wcs.wcs.ctype[axis]
        newwcs.wcs.crval[axis + 2] = griddata.grid_wcs.wcs.crval[axis]
        newwcs.wcs.crpix[axis + 2] = griddata.grid_wcs.wcs.crpix[axis]
        newwcs.wcs.cdelt[axis + 2] = griddata.grid_wcs.wcs.cdelt[axis]
    
    newwcs.wcs.ctype[0] = 'UU'
    newwcs.wcs.ctype[1] = 'VV'
    newwcs.wcs.ctype[2] = 'DUU'
    newwcs.wcs.ctype[3] = 'DVV'
    newwcs.wcs.cdelt[0] *= oversampling
    newwcs.wcs.cdelt[1] *= oversampling
    
    newwcs.wcs.crpix[0] = support / 2
    newwcs.wcs.crpix[1] = support / 2
    newwcs.wcs.crpix[2] = oversampling / 2 + 1
    newwcs.wcs.crpix[3] = oversampling / 2 + 1

    for axis in range(2):
        newwcs.wcs.cdelt[axis] = oversampling * griddata.grid_wcs.wcs.cdelt[axis]
        newwcs.wcs.cdelt[axis + 2] = griddata.grid_wcs.wcs.cdelt[axis]
        newwcs.wcs.crpix[axis] = support / 2 + 1
    
    return create_convolutionfunction_from_array(data=newgrid, grid_wcs=newwcs,
                                                 projection_wcs=griddata.projection_wcs,
                                                 polarisation_frame=griddata.polarisation_frame)