# Tim Cornwell realtimcornwell@gmail.com
#
"""FFT support functions

All grids and images are considered quadratic and centered around
`N//2`, where `N` is the pixel width/height. This means that `N//2` is
the zero frequency for FFT purposes, as is convention. Note that this
means that for even `N` the grid is not symmetrical, which means that
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
import scipy.special


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
                     pad_width=2 * [(N // 2 - N0 // 2, (N + 1) // 2 - (N0 + 1) // 2)],
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
    my = Na // 2 - Qpx * (N // 2) - yf
    mx = Na // 2 - Qpx * (N // 2) - xf
    assert mx >= 0 and my >= 0
    # Extract every Qpx-th pixel
    mid = a[my: my + Qpx * N: Qpx,
          mx: mx + Qpx * N: Qpx]
    # normalise
    return Qpx * Qpx * mid