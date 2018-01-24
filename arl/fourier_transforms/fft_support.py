""" FFT support functions

"""

import numpy


def fft(a):
    """ Fourier transformation from image to grid space
    
    .. note::
    
        If there are four axes then the last outer axes are not transformed

    :param a: image in `lm` coordinate space
    :return: `uv` grid
    """
    if (len(a.shape) == 4):
        return numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(a, axes=[2, 3])), axes=[2, 3])
    else:
        return numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(a)))


def ifft(a):
    """ Fourier transformation from grid to image space

    .. note::
    
        If there are four axes then the last outer axes are not transformed

    :param a: `uv` grid to transform
    :return: an image in `lm` coordinate space
    """
    if (len(a.shape) == 4):
        return numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.ifftshift(a, axes=[2, 3])), axes=[2, 3])
    else:
        return numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.ifftshift(a)))


def pad_mid(ff, npixel):
    """
    Pad a far field image with zeroes to make it the given size.

    Effectively as if we were multiplying with a box function of the
    original field's size, which is equivalent to a convolution with a
    sinc pattern in the uv-grid.

    .. note::
    
        Only the two innermost axes are transformed

        This function does not handle odd-sized dimensions properly

    :param ff: The input far field. Should be smaller than npixelxnpixel.

    :param npixel:  The desired far field size

    """
    ny, nx = ff.shape[-2:]
    cx = nx // 2
    cy = ny // 2
    if npixel == nx:
        return ff
    assert npixel > nx and npixel > ny
    pw = [(0, 0)] * (ff.ndim-2)  + [(npixel // 2 - cy, npixel // 2 - cy),
                                    (npixel // 2 - cx, npixel // 2 - cx)]
    return numpy.pad(ff,
                     pad_width=pw,
                     mode='constant',
                     constant_values=0.0)


def extract_mid(a, npixel):
    """
    Extract a section from middle of a map

    Suitable for zero frequencies at npixel/2. This is the reverse
    operation to pad.

    .. note::
    
        Only the two innermost axes are transformed

    :param npixel: desired size of the section to extract
    :param a: grid from which to extract
    """
    ny, nx = a.shape[-2:]
    cx = nx // 2
    cy = ny // 2
    s = npixel // 2
    if npixel % 2 != 0:
        return a[..., cx - s:cx + s + 1, cy - s:cy + s + 1]
    else:
        return a[..., cx - s:cx + s, cy - s:cy + s]



def extract_oversampled(a, xf, yf, kernel_oversampling, kernelwidth):
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
    :param kernelwidth: size of section
    """
    
    assert 0 <= xf < kernel_oversampling
    assert 0 <= yf < kernel_oversampling
    # Determine start offset.
    npixela = a.shape[0]
    my = npixela // 2 - kernel_oversampling * (kernelwidth // 2) - yf
    mx = npixela // 2 - kernel_oversampling * (kernelwidth // 2) - xf
    assert mx >= 0 and my >= 0, "mx %d and my %d" % (mx, my)
    # Extract every kernel_oversampling-th pixel
    mid = a[my: my + kernel_oversampling * kernelwidth: kernel_oversampling,
            mx: mx + kernel_oversampling * kernelwidth: kernel_oversampling]
    # normalise
    return kernel_oversampling * kernel_oversampling * mid
