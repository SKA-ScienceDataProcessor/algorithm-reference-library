# Tim Cornwell realtimcornwell@gmail.com
#
"""FFT support functions

All grids and images are considered quadratic and centered around
`npixel//2`, where `npixel` is the pixel width/height. This means that `npixel//2` is
the zero frequency for FFT purposes, as is convention. Note that this
means that for even `npixel` the grid is not symmetrical, which means that
e.g. for convolution kernels odd image sizes are preferred.

This is implemented for reference in
`coordinates`/`coordinates2`. Some noteworthy properties:
- `ceil(field_of_view * lam)` gives the image size `npixel` in pixels
- `lam * coordinates2(npixel)` yields the `u,v` grid coordinate system
- `field_of_view * coordinates2(npixel)` yields the `l,m` image coordinate system
   (radians, roughly)
   
"""

from __future__ import division

import numpy


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


def pad_mid(ff, npixel):
    """
    Pad a far field image with zeroes to make it the given size.

    Effectively as if we were multiplying with a box function of the
    original field's size, which is equivalent to a convolution with a
    sinc pattern in the uv-grid.

    :param ff: The input far field. Should be smaller than NxN.
    :param npixel:  The desired far field size

    """
    
    npixel0, npixel0w = ff.shape
    if npixel == npixel0:
        return ff
    assert npixel > npixel0 == npixel0w
    return numpy.pad(ff,
                     pad_width=2 * [(npixel // 2 - npixel0 // 2, (npixel + 1) // 2 - (npixel0 + 1) // 2)],
                     mode='constant',
                     constant_values=0.0)


def extract_mid(a, npixel):
    """
    Extract a section from middle of a map

    Suitable for zero frequencies at npixel/2. This is the reverse
    operation to pad.

    :param npixel:
    :param a: grid from which to extract
    """
    cx = a.shape[0] // 2
    cy = a.shape[1] // 2
    s = npixel // 2
    if npixel % 2 != 0:
        return a[cx - s:cx + s + 1, cy - s:cy + s + 1]
    else:
        return a[cx - s:cx + s, cy - s:cy + s]


def extract_oversampled(a, xf, yf, kernel_oversampling, npixel):
    """
    Extract the (xf-th,yf-th) w-kernel from the oversampled parent

    Offsets are suitable for correcting of fractional coordinates,
    e.g. an offset of (xf,yf) results in the kernel for an (-xf,-yf)
    sub-grid offset.

    We do not want to make assumptions about the source grid's symmetry
    here, which means that the grid's side length must be at least
    kernel_oversampling*(npixel+2) to contain enough information in all circumstances

    :param xf:
    :param yf:
    :param a: grid from which to extract
    :param kernel_oversampling: oversampling factor
    :param npixel: size of section
    """
    
    assert 0 <= xf < kernel_oversampling
    assert 0 <= yf < kernel_oversampling
    # Determine start offset.
    npixela = a.shape[0]
    my = npixela // 2 - kernel_oversampling * (npixel // 2) - yf
    mx = npixela // 2 - kernel_oversampling * (npixel // 2) - xf
    assert mx >= 0 and my >= 0, "mx %d and my %d" % (mx, my)
    # Extract every kernel_oversampling-th pixel
    mid = a[my: my + kernel_oversampling * npixel: kernel_oversampling,
          mx: mx + kernel_oversampling * npixel: kernel_oversampling]
    # normalise
    return kernel_oversampling * kernel_oversampling * mid
