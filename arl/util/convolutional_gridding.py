# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
#
# Synthesise and Image interferometer data
"""Convolutional gridding support functions

All functions that involve convolutional gridding are kept here.
"""

from __future__ import division

import logging

import scipy.special
from arl.fourier_transforms.fft_support import *

from data.parameters import get_parameter

log = logging.getLogger("convolutional.gridding")


def _coordinateBounds(npixel):
    r""" Returns lowest and highest coordinates of an image/grid given:

    1. Step size is :math:`1/npixel`:

       .. math:: \frac{high-low}{npixel-1} = \frac{1}{npixel}

    2. The coordinate :math:`\lfloor npixel/2\rfloor` falls exactly on zero:

       .. math:: low + \left\lfloor\frac{npixel}{2}\right\rfloor * (high-low) = 0

    This is the coordinate system for shifted FFTs.
    """
    if npixel % 2 == 0:
        return -0.5, 0.5 * (npixel - 2) / npixel
    else:
        return -0.5 * (npixel - 1) / npixel, 0.5 * (npixel - 1) / npixel


def _coordinates(npixel):
    """ 1D array which spans [-.5,.5[ with 0 at position npixel/2
    
    """
    low, high = _coordinateBounds(npixel)
    return numpy.mgrid[low:high:(npixel * 1j)]


def _coordinates2(npixel):
    """Two dimensional grids of coordinates spanning -1 to 1 in each dimension

    1. a step size of 2/npixel and
    2. (0,0) at pixel (floor(n/2),floor(n/2))
    """
    low, high = _coordinateBounds(npixel)
    return numpy.mgrid[low:high:(npixel * 1j), low:high:(npixel * 1j)]


def anti_aliasing_function(shape, m, c):
    """
    Compute the prolate spheroidal anti-aliasing function

    See VLA Scientific Memoranda 129, 131, 132
    :param shape: (height, width) pair
    :param m: mode parameter
    :param c: spheroidal parameter
    """
    
    # 2D Prolate spheroidal angular function is separable
    sy, sx = [scipy.special.pro_ang1(m, m, c, _coordinates(npixel))[0] for npixel in shape]
    return numpy.outer(sy, sx)


def _w_kernel_function(npixel, field_of_view, w):
    """
    W beam, the fresnel diffraction pattern arising from non-coplanar baselines

    :param npixel: Size of the grid in pixels
    :param field_of_view: Field of view
    :param w: Baseline distance to the projection plane
    :returns: npixel x npixel array with the far field
    """
    
    m, l = _coordinates2(npixel) * field_of_view
    r2 = l ** 2 + m ** 2
    assert numpy.array(r2 < 1.0).all(), \
        "Error in image coordinate system: field_of_view %f, npixel %f,l %s, m %s" % (field_of_view, npixel, l, m)
    ph = w * (1 - numpy.sqrt(1.0 - r2))
    cp = numpy.exp(2j * numpy.pi * ph)
    return cp


def kernel_coordinates(npixel, field_of_view, dl=0, dm=0, transform_matrix=None):
    """
    Returns (l,m) coordinates for generation of kernels in a far-field of the given size.

    If coordinate transformations are passed, they must be inverse to
    the transformations applied to the visibilities using
    visibility_shift/uvw_transform.

    :param field_of_view:
    :param npixel: Desired far-field size
    :param dl: Pattern horizontal shift (see visibility_shift)
    :param dm: Pattern vertical shift (see visibility_shift)
    :param transformmatrix: Pattern transformation matrix (see uvw_transform)
    :returns: Pair of (m,l) coordinates
    """
    
    m, l = _coordinates2(npixel) * field_of_view
    if transform_matrix is not None:
        l, m = transform_matrix[0, 0] * l + transform_matrix[1, 0] * m, transform_matrix[0, 1] * l \
               + transform_matrix[1, 1] * m
    return m + dm, l + dl


def _kernel_oversample(ff, npixel, kernel_oversampling, s):
    """ Takes a farfield pattern and creates an oversampled convolution function.

    If the far field size is smaller than npixel*kernel_oversampling, we will pad it. This
    essentially means we apply a sinc anti-aliasing kernel by default.

    :param ff: Far field pattern
    :param npixel: Image size without oversampling
    :param kernel_oversampling: Factor to oversample by -- there will be kernel_oversampling x kernel_oversampling
    convolution functions
    :param s: Size of convolution function to extract
    :returns: Numpy array of shape [ov, ou, v, u], e.g. with sub-pixel offsets as the outer coordinates.
    """
    
    # Pad the far field to the required pixel size
    padff = pad_mid(ff, npixel * kernel_oversampling)
    
    # Obtain oversampled uv-grid
    af = ifft(padff)
    
    # Extract kernels
    res = [[extract_oversampled(af, x, y, kernel_oversampling, s)
            for x in range(kernel_oversampling)]
           for y in range(kernel_oversampling)]
    return numpy.array(res)


def _w_kernel(field_of_view, w, npixel_farfield, npixel_kernel, kernel_oversampling):
    """ The middle s pixels of W convolution kernel. (W-KERNel-Aperture-Function)

    :param field_of_view: Field of view (directional cosines)
    :param w: Baseline distance to the projection plane
    :param npixel_farfield: Far field size. Must be at least npixel_kernel+1 if kernel_oversampling > 1, otherwise npixel_kernel.
    :param npixel_kernel: Size of convolution function to extract
    :param kernel_oversampling: Oversampling, pixels will be kernel_oversampling smaller in aperture
      plane than required to minimially sample field_of_view.

    :returns: [kernel_oversampling,kernel_oversampling,s,s] shaped oversampled convolution kernels
    """
    assert npixel_farfield > npixel_kernel or (npixel_farfield == npixel_kernel and kernel_oversampling == 1)
    return _kernel_oversample(_w_kernel_function(npixel_farfield, field_of_view, w), npixel_farfield,
                              kernel_oversampling, npixel_kernel)


def _frac_coord(npixel, kernel_oversampling, p):
    """ Compute whole and fractional parts of coordinates, rounded to
    kernel_oversampling-th fraction of pixel size

    The fractional values are rounded to nearest 1/kernel_oversampling pixel value. At
    fractional values greater than (kernel_oversampling-0.5)/kernel_oversampling coordinates are
    rounded to next integer index.

    :param npixel: Number of pixels in total
    :param kernel_oversampling: Fractional values to round to
    :param p: Coordinate in range [-.5,.5[
    """
    assert numpy.array(p >= -0.5).all() and numpy.array(p < 0.5).all(), "uv overflows grid uv= %s" % str(p)
    x = npixel // 2 + p * npixel
    flx = numpy.floor(x + 0.5 / kernel_oversampling)
    fracx = numpy.around((x - flx) * kernel_oversampling)
    return flx.astype(int), fracx.astype(int)


def _frac_coords(shape, kernel_oversampling, xycoords):
    """Compute grid coordinates and fractional values for convolutional gridding

    :param shape: (height,width) grid shape
    :param kernel_oversampling: Oversampling factor
    :param xycoords: array of (x,y) coordinates in range [-.5,.5[
    """
    _, _, h, w = shape  # NB order (height,width) to match numpy!
    y, yf = _frac_coord(h, kernel_oversampling, xycoords[:, 1])
    x, xf = _frac_coord(w, kernel_oversampling, xycoords[:, 0])
    return x, xf, y, yf


def fixed_kernel_degrid(kernel, uvgrid, uv, uvscale):
    """Convolutional degridding with frequency and polarisation independent

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param kernel: Oversampled convolution kernel
    :param uvgrid:   The uv plane to de-grid from
    :param uv: fractional uv coordinates in range[-0.5,0.5[
    :param uvscale: scaling for each channel
    :returns: Array of visibilities.
    """
    kernel_oversampling, _, gh, gw = kernel.shape
    nchan, npol, ny, nx = uvgrid.shape
    nvis, _ = uv.shape
    vis = numpy.zeros([nvis, nchan, npol], dtype='complex')
    wt = numpy.zeros([nvis, nchan, npol])
    for chan in range(nchan):
        coords = _frac_coords(uvgrid.shape, kernel_oversampling, uv * uvscale[chan])
        for pol in range(npol):
            vis[..., chan, pol] = [
                numpy.sum(uvgrid[chan, pol, y - gh // 2: y + (gh + 1) // 2, x - gw // 2: x + (gw + 1) // 2]
                          * kernel[yf, xf])
                for x, xf, y, yf in zip(*coords)
                ]
            wt[..., chan, pol] = [
                numpy.sum(kernel[yf, xf].real)
                for x, xf, y, yf in zip(*coords)
                ]
    vis[numpy.where(wt > 0)] = vis[numpy.where(wt > 0)] / wt[numpy.where(wt > 0)]
    vis[numpy.where(wt < 0)] = 0.0
    return numpy.array(vis)


def fixed_kernel_grid(kernel, uvgrid, uv, uvscale, vis, visweights):
    """Grid after convolving with frequency and polarisation independent gcf

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param kernel: Oversampled convolution kernel
    :param uvgrid: Grid to add to
    :param uv: UVW positions
    :param vis: Visibility values
    :param vis: Visibility weights
    """
    
    kernel_oversampling, _, gh, gw = kernel.shape
    nchan, npol, ny, nx = uvgrid.shape
    for chan in range(nchan):
        coords = _frac_coords(uvgrid.shape, kernel_oversampling, uv * uvscale[chan])
        for pol in range(npol):
            for v, x, xf, y, yf in zip(visweights[..., chan, pol] * vis[..., chan, pol], *coords):
                uvgrid[chan, pol, (y - gh // 2):(y + (gh + 1) // 2), (x - gw // 2):(x + (gw + 1) // 2)] \
                    += kernel[yf, xf] * v
    
    return uvgrid


def weight_gridding(shape, uv, uvscale, visweights, params):
    """Reweight data using one of a number of algorithms

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param shape:
    :param uv: UVW positions
    :param uvscale: Scaling to uv (per channel)
    :param vis: Visibility values
    :param visweights: Visibility weights
    """
    weighting = get_parameter(params, 'weighting', 'uniform')
    if weighting == 'uniform':
        log.debug("convolutional_gridding.weight_gridding: Performing uniform weighting")
        wtsgrid = numpy.zeros(shape)
        nchan, npol, ny, nx = shape
        # Add all visibility points to a float grid
        for chan in range(nchan):
            coords = _frac_coords(shape, 1.0, uv * uvscale[chan])
            for pol in range(npol):
                for wt, x, _, y, _ in zip(visweights[..., chan, pol], *coords):
                    wtsgrid[chan, pol, y, x] += wt
        # Normalise each visibility weight to sum to one in a grid cell
        newvisweights = numpy.zeros_like(visweights)
        for chan in range(nchan):
            coords = _frac_coords(shape, 1.0, uv * uvscale[chan])
            for pol in range(npol):
                newvisweights[..., chan, pol] = [wt / wtsgrid[chan, pol, y, x]
                                                 for wt, x, _, y, _ in zip(visweights[..., chan, pol], *coords)
                                                 ]
        return newvisweights
    else:
        return visweights
