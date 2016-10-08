# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
#
# Synthesise and Image interferometer data
"""kernel support functions

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

import numpy
from functools import lru_cache
import scipy.special

def coordinateBounds(N):
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

def coordinates(N):
    """ 1D array which spans [-.5,.5[ with 0 at position N/2
    
    """
    low, high = coordinateBounds(N)
    return numpy.mgrid[low:high:(N * 1j)]


def coordinates2(N):
    """Two dimensional grids of coordinates spanning -1 to 1 in each dimension

    1. a step size of 2/N and
    2. (0,0) at pixel (floor(n/2),floor(n/2))
    """
    low, high = coordinateBounds(N)
    return numpy.mgrid[low:high:(N * 1j), low:high:(N * 1j)]


def fft(a):
    """ Fourier transformation from image to grid space

    :param a: image in `lm` coordinate space
    :returns: `uv` grid
    """
    return numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(a)))


def ifft(a):
    """ Fourier transformation from grid to image space

    :param a: `uv` grid to transform
    :returns: an image in `lm` coordinate space
    """
    return numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.ifftshift(a)))


def pad_mid(ff, N):
    """
    Pad a far field image with zeroes to make it the given size.

    Effectively as if we were multiplying with a box function of the
    original field's size, which is equivalent to a convolution with a
    sinc pattern in the uv-grid.

    :param ff: The input far field. Should be smaller than NxN.
    :param N:  The desired far field size

    """

    N0, N0w = ff.shape
    if N == N0: return ff
    assert N > N0 and N0 == N0w
    return numpy.pad(ff,
                     pad_width=2*[(N//2-N0//2, (N+1)//2-(N0+1)//2)],
                     mode='constant',
                     constant_values=0.0)


def extract_mid(a, N):
    """
    Extract a section from middle of a map

    Suitable for zero frequencies at N/2. This is the reverse
    operation to pad.

    :param a: grid from which to extract
    :param s: size of section
    """
    cx = a.shape[0] // 2
    cy = a.shape[1] // 2
    s = N // 2
    if N % 2 != 0:
        return a[cx - s:cx + s + 1, cy - s:cy + s + 1]
    else:
        return a[cx - s:cx + s, cy - s:cy + s]
    

def extract_oversampled(a, xf, yf, Qpx, N):
    """
    Extract the (xf-th,yf-th) w-kernel from the oversampled parent

    Offsets are suitable for correcting of fractional coordinates,
    e.g. an offset of (xf,yf) results in the kernel for an (-xf,-yf)
    sub-grid offset.

    We do not want to make assumptions about the source grid's symmetry
    here, which means that the grid's side length must be at least
    Qpx*(N+2) to contain enough information in all circumstances

    :param a: grid from which to extract
    :param ox: x offset
    :param oy: y offset
    :param Qpx: oversampling factor
    :param N: size of section
    """

    assert xf >= 0 and xf < Qpx
    assert yf >= 0 and yf < Qpx
    # Determine start offset.
    Na = a.shape[0]
    my = Na//2 - Qpx*(N//2) - yf
    mx = Na//2 - Qpx*(N//2) - xf
    assert mx >= 0 and my >= 0
    # Extract every Qpx-th pixel
    mid = a[my : my+Qpx*N : Qpx,
            mx : mx+Qpx*N : Qpx]
    # normalise
    return Qpx * Qpx * mid


def anti_aliasing_function(shape, m, c):
    """
    Compute the prolate spheroidal anti-aliasing function

    See VLA Scientific Memoranda 129, 131, 132
    :param shape: (height, width) pair
    :param m: mode parameter
    :param c: spheroidal parameter
    """

    # 2D Prolate spheroidal angular function is separable
    sy, sx = [ scipy.special.pro_ang1(m, m, c, coordinates(N))[0] for N in shape ]
    return numpy.outer(sy, sx)


def w_kernel_function(N, field_of_view, w):
    """
    W beam, the fresnel diffraction pattern arising from non-coplanar baselines

    :param N: Size of the grid in pixels
    :param field_of_view: Field of view
    :param w: Baseline distance to the projection plane
    :returns: N x N array with the far field
    """

    m, l = coordinates2(N) * field_of_view
    r2 = l**2 + m**2
    assert numpy.all(r2 < 1.0), "Error in image coordinate system: theta %f, N %f,l %s, m %s" % (field_of_view, N, l, m)
    ph = w * (1 - numpy.sqrt(1.0 - r2))
    cp = numpy.exp(2j * numpy.pi * ph)
    return cp

def kernel_oversample(ff, N, Qpx, s):
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
    padff = pad_mid(ff, N*Qpx)

    # Obtain oversampled uv-grid
    af = ifft(padff)

    # Extract kernels
    res = [[extract_oversampled(af, x, y, Qpx, s) for x in range(Qpx)] for y in range(Qpx)]
    return numpy.array(res)


def w_kernel(field_of_view, w, NpixFF, NpixKern, Qpx):
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
    return kernel_oversample(w_kernel_function(NpixFF, field_of_view, w), NpixFF, Qpx, NpixKern)