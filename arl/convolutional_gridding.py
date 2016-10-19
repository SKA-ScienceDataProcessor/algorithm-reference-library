# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
#
# Synthesise and Image interferometer data
"""Convolutional gridding support functions

Parameter name meanings:

- p: The uvw coordinates [*,3] (m)
- v: The Visibility values [*] (Jy)
- field_of_view: Width of the field of view to be synthesised, as directional
  cosines (approximately radians)
- lam: Width of the uv-plane (in wavelengths). Controls resolution of the
  images.
- Qpx: Oversampling of pixels by the convolution kernels -- there are
  (Qpx x Qpx) convolution kernels per pixels to account for fractional
  pixel values.

All grids and images are considered quadratic and centered around
`N//2`, where `N` is the pixel width/height. This means that `N//2` is
the zero frequency for FFT purposes, as is convention. Note that this
means that for even `N` the grid is not symetrical, which means that
e.g. for convolution kernels odd image sizes are preferred.

This is implemented for reference in
`coordinates`/`coordinates2`. Some noteworthy properties:
- `ceil(field_of_view * lam)` gives the image size `N` in pixels
- `lam * coordinates2(N)` yields the `u,v` grid coordinate system
- `field_of_view * coordinates2(N)` yields the `l,m` image coordinate system
   (radians, roughly)
   
"""

from __future__ import division

import scipy.special

from arl.fft_support import *


def _coordinateBounds(N):
    r""" Returns lowest and highest coordinates of an image/grid given:

    1. Step size is :math:`1/N`:

       .. math:: \frac{high-low}{N-1} = \frac{1}{N}

    2. The coordinate :math:`\lfloor N/2\rfloor` falls exactly on zero:

       .. math:: low + \left\lfloor\frac{N}{2}\right\rfloor * (high-low) = 0

    This is the coordinate system for shifted FFTs.
    """
    if N % 2 == 0:
        return -0.5, 0.5 * (N - 2) / N
    else:
        return -0.5 * (N - 1) / N, 0.5 * (N - 1) / N


def _coordinates(N):
    """ 1D array which spans [-.5,.5[ with 0 at position N/2
    
    """
    low, high = _coordinateBounds(N)
    return numpy.mgrid[low:high:(N * 1j)]


def _coordinates2(N):
    """Two dimensional grids of coordinates spanning -1 to 1 in each dimension

    1. a step size of 2/N and
    2. (0,0) at pixel (floor(n/2),floor(n/2))
    """
    low, high = _coordinateBounds(N)
    return numpy.mgrid[low:high:(N * 1j), low:high:(N * 1j)]


def anti_aliasing_function(shape, m, c):
    """
    Compute the prolate spheroidal anti-aliasing function

    See VLA Scientific Memoranda 129, 131, 132
    :param shape: (height, width) pair
    :param m: mode parameter
    :param c: spheroidal parameter
    """
    
    # 2D Prolate spheroidal angular function is separable
    sy, sx = [scipy.special.pro_ang1(m, m, c, _coordinates(N))[0] for N in shape]
    return numpy.outer(sy, sx)


def _w_kernel_function(N, field_of_view, w):
    """
    W beam, the fresnel diffraction pattern arising from non-coplanar baselines

    :param N: Size of the grid in pixels
    :param field_of_view: Field of view
    :param w: Baseline distance to the projection plane
    :returns: N x N array with the far field
    """
    
    m, l = _coordinates2(N) * field_of_view
    r2 = l ** 2 + m ** 2
    assert numpy.all(r2 < 1.0), \
        "Error in image coordinate system: field_of_view %f, N %f,l %s, m %s" % (field_of_view, N, l, m)
    ph = w * (1 - numpy.sqrt(1.0 - r2))
    cp = numpy.exp(2j * numpy.pi * ph)
    return cp


def kernel_coordinates(N, theta, dl=0, dm=0, T=None):
    """
    Returns (l,m) coordinates for generation of kernels
    in a far-field of the given size.

    If coordinate transformations are passed, they must be inverse to
    the transformations applied to the visibilities using
    visibility_shift/uvw_transform.

    :param theta:
    :param N: Desired far-field size
    :param dl: Pattern horizontal shift (see visibility_shift)
    :param dm: Pattern vertical shift (see visibility_shift)
    :param T: Pattern transformation matrix (see uvw_transform)
    :returns: Pair of (m,l) coordinates
    """
    
    m, l = _coordinates2(N) * theta
    if T is not None:
        l, m = T[0, 0] * l + T[1, 0] * m, T[0, 1] * l + T[1, 1] * m
    return m + dm, l + dl


def _kernel_oversample(ff, N, Qpx, s):
    """
    Takes a farfield pattern and creates an oversampled convolution
    function.

    If the far field size is smaller than N*Qpx, we will pad it. This
    essentially means we apply a sinc anti-aliasing kernel by default.

    :param ff: Far field pattern
    :param N:  Image size without oversampling
    :param Qpx: Factor to oversample by -- there will be Qpx x Qpx convolution arl
    :param s: Size of convolution function to extract
    :returns: Numpy array of shape [ov, ou, v, u], e.g. with sub-pixel
      offsets as the outer coordinates.
    """
    
    # Pad the far field to the required pixel size
    padff = pad_mid(ff, N * Qpx)
    
    # Obtain oversampled uv-grid
    af = ifft(padff)
    
    # Extract kernels
    res = [[extract_oversampled(af, x, y, Qpx, s) for x in range(Qpx)] for y in range(Qpx)]
    return numpy.array(res)


def _w_kernel(field_of_view, w, NpixFF, NpixKern, Qpx):
    """
    The middle s pixels of W convolution kernel. (W-KERNel-Aperture-Function)

    :param field_of_view: Field of view (directional cosines)
    :param w: Baseline distance to the projection plane
    :param NpixFF: Far field size. Must be at least NpixKern+1 if Qpx > 1, otherwise NpixKern.
    :param NpixKern: Size of convolution function to extract
    :param Qpx: Oversampling, pixels will be Qpx smaller in aperture
      plane than required to minimially sample field_of_view.

    :returns: [Qpx,Qpx,s,s] shaped oversampled convolution kernels
    """
    assert NpixFF > NpixKern or (NpixFF == NpixKern and Qpx == 1)
    return _kernel_oversample(_w_kernel_function(NpixFF, field_of_view, w), NpixFF, Qpx, NpixKern)


def _frac_coord(N, Qpx, p):
    """
    Compute whole and fractional parts of coordinates, rounded to
    Qpx-th fraction of pixel size

    The fractional values are rounded to nearest 1/Qpx pixel value. At
    fractional values greater than (Qpx-0.5)/Qpx coordinates are
    roundeded to next integer index.

    :param N: Number of pixels in total
    :param Qpx: Fractional values to round to
    :param p: Coordinate in range [-.5,.5[
    """
    assert (p >= -0.5).all() and (p < 0.5).all()
    x = N // 2 + p * N
    flx = numpy.floor(x + 0.5 / Qpx)
    fracx = numpy.around((x - flx) * Qpx)
    return flx.astype(int), fracx.astype(int)


def _frac_coords(shape, Qpx, xycoords):
    """Compute grid coordinates and fractional values for convolutional
    gridding

    :param shape: (height,width) grid shape
    :param Qpx: Oversampling factor
    :param xycoords: array of (x,y) coordinates in range [-.5,.5[
    """
    h, w = shape  # NB order (height,width) to match numpy!
    x, xf = _frac_coord(w, Qpx, xycoords[:, 0])
    y, yf = _frac_coord(h, Qpx, xycoords[:, 1])
    return x, xf, y, yf


def convolutional_degrid(gcf, uvgrid, uv):
    """Convolutional degridding with frequency and polarisation independent

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param uv:
    :param gcf: Oversampled convolution kernel
    :param uvgrid:   The uv plane to de-grid from
    :returns: Array of visibilities.
    """
    Qpx, _, gh, gw = gcf.shape
    coords = _frac_coords(uvgrid.shape, Qpx, uv)
    vis = [
        numpy.sum(uvgrid[..., y - gh // 2: y + (gh + 1) // 2, x - gw // 2: x + (gw + 1) // 2] * gcf[yf, xf])
        for x, xf, y, yf in zip(*coords)
        ]
    return numpy.array(vis)


def convolutional_grid(gcf, uvgrid, uv, vis):
    """Grid after convolving with frequency and polarisation independent gcf

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param gcf: Oversampled convolution kernel
    :param uvgrid: Grid to add to
    :param uv: UVW positions
    :param vis: Visibility values
    """
    
    Qpx, _, gh, gw = gcf.shape
    coords = _frac_coords(uvgrid.shape, Qpx, uv)
    for vis, x, xf, y, yf in zip(vis, *coords):
        uvgrid[..., y - gh // 2: y + (gh + 1) // 2, x - gw // 2: x + (gw + 1) // 2] += gcf[yf, xf] * vis
