#
"""
Functions that define and manipulate GridData.

The griddata has axes [chan, pol, z, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes.

GridData can be used to hold the Fourier transform of an Image or gridded visibilities. In addition, the convolution
function can be stored in a GridData, most probably with finer spatial sampling.


"""
import copy
import logging

import numpy
from astropy.wcs import WCS

from data_models.memory_data_models import QA


from data_models.memory_data_models import GridData
from data_models.polarisation import PolarisationFrame
from libs.fourier_transforms.fft_support import ifft, fft
from libs.image.operations import create_image_from_array

log = logging.getLogger(__name__)


def griddata_sizeof(gd: GridData):
    """ Return size in GB
    """
    return gd.size()


def create_griddata_from_array(data: numpy.array, grid_wcs: WCS, projection_wcs: WCS,
                               polarisation_frame: PolarisationFrame) -> GridData:
    """ Create a griddata from an array and wcs's
    
    The griddata has axes [chan, pol, z, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
    order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes
    
    Griddata holds the original sky plane projection in the projection_wcs.

    :param data: Numpy.array
    :param grid_wcs: Grid world coordinate system
    :param projection_wcs: Projection world coordinate system
    :param polarisation_frame: Polarisation Frame
    :return: GridData
    
    """
    fgriddata = GridData()
    fgriddata.polarisation_frame = polarisation_frame
    
    fgriddata.data = data
    fgriddata.grid_wcs = grid_wcs.deepcopy()
    fgriddata.projection_wcs = projection_wcs.deepcopy()
    
    if griddata_sizeof(fgriddata) >= 1.0:
        log.debug("create_griddata_from_array: created %s image of shape %s, size %.3f (GB)" %
                  (fgriddata.data.dtype, str(fgriddata.shape), griddata_sizeof(fgriddata)))
    
    assert isinstance(fgriddata, GridData), "Type is %s" % type(fgriddata)
    return fgriddata


def create_griddata_from_image(im, nw=1, wstep=1e15):
    """ Create a GridData from an image

    :param im: Image
    :param nw: Number of w planes
    :param wstep: Increment in w
    :return: GridData
    """
    assert len(im.shape) == 4
    assert im.wcs.wcs.ctype[0] == 'RA---SIN'
    assert im.wcs.wcs.ctype[1] == 'DEC--SIN'
    
    d2r = numpy.pi / 180.0
    projection_wcs = copy.deepcopy(im.wcs)
    
    # WCS Coords are [x, y, z, pol, chan] where x, y, z are spatial axes in real space or Fourier space
    # Array Coords are [chan, pol, z, y, x] where x, y, z are spatial axes in real space or Fourier space
    grid_wcs = WCS(naxis=5)
    
    grid_wcs.wcs.axis_types[0] = 0
    grid_wcs.wcs.axis_types[1] = 0
    grid_wcs.wcs.axis_types[2] = 0
    grid_wcs.wcs.axis_types[3] = im.wcs.wcs.axis_types[2]
    grid_wcs.wcs.axis_types[4] = im.wcs.wcs.axis_types[3]
    
    grid_wcs.wcs.crval[0] = 0.0
    grid_wcs.wcs.crval[1] = 0.0
    grid_wcs.wcs.crval[2] = 0.0
    grid_wcs.wcs.crval[3] = im.wcs.wcs.crval[2]
    grid_wcs.wcs.crval[4] = im.wcs.wcs.crval[3]
    
    grid_wcs.wcs.crpix[0] = im.shape[3] // 2 + 1
    grid_wcs.wcs.crpix[1] = im.shape[2] // 2 + 1
    grid_wcs.wcs.crpix[2] = nw // 2
    grid_wcs.wcs.crpix[3] = im.wcs.wcs.crpix[2]
    grid_wcs.wcs.crpix[4] = im.wcs.wcs.crpix[3]
    
    grid_wcs.wcs.ctype[0] = 'UU'
    grid_wcs.wcs.ctype[1] = 'VV'
    grid_wcs.wcs.ctype[2] = 'WW'
    grid_wcs.wcs.ctype[3] = im.wcs.wcs.ctype[2]
    grid_wcs.wcs.ctype[4] = im.wcs.wcs.ctype[3]
    
    grid_wcs.wcs.cdelt[0] = 1.0 / (im.shape[3] * d2r * im.wcs.wcs.cdelt[0])
    grid_wcs.wcs.cdelt[1] = 1.0 / (im.shape[2] * d2r * im.wcs.wcs.cdelt[1])
    grid_wcs.wcs.cdelt[2] = wstep
    grid_wcs.wcs.cdelt[3] = im.wcs.wcs.cdelt[2]
    grid_wcs.wcs.cdelt[4] = im.wcs.wcs.cdelt[3]
    
    nchan, npol, ny, nx = im.shape
    grid_data = numpy.zeros([nchan, npol, nw, ny, nx], dtype='complex')
    
    return create_griddata_from_array(grid_data, grid_wcs=grid_wcs,
                                      projection_wcs=projection_wcs,
                                      polarisation_frame=im.polarisation_frame)


def fft_image_to_griddata(im):
    """ FFT an image to GridData

    :param im:
    :return: GridData
    """
    gd = create_griddata_from_image(im)
    assert gd.shape[2] == 1, "Multiple z planes not supported"
    gd.data[:, :, 0, ...] = ifft(im.data[:, :, :, :].data)
    
    return gd


def fft_griddata_to_image(gd: GridData):
    """ FFT GridData to an Image, transform WCS as well

    :param gd: GridData model
    :return: Image
    """
    assert len(gd.shape) == 5
    assert gd.grid_wcs.wcs.ctype[0] == 'UU'
    assert gd.grid_wcs.wcs.ctype[1] == 'VV'
    wcs = gd.projection_wcs.deepcopy()
    assert gd.shape[2] == 1, "Multiple z planes not supported"
    image_data = fft(gd.data[:, :, 0, ...].astype('complex'))
    return create_image_from_array(image_data, wcs=wcs, polarisation_frame=gd.polarisation_frame)


def convert_griddata_to_image(gd):
    """ Convert griddata to an image
    
    :param gd:
    :return:
    """
    return create_image_from_array(gd.data, gd.grid_wcs, gd.polarisation_frame)


def qa_griddata(gd, context="") -> QA:
    """Assess the quality of a griddata

    :param gd:
    :return: QA
    """
    assert isinstance(gd, GridData), gd
    data = {'shape': str(gd.data.shape),
            'max': numpy.max(gd.data),
            'min': numpy.min(gd.data),
            'rms': numpy.std(gd.data),
            'sum': numpy.sum(gd.data),
            'medianabs': numpy.median(numpy.abs(gd.data)),
            'median': numpy.median(gd.data)}
    
    qa = QA(origin="qa_image", data=data, context=context)
    return qa


