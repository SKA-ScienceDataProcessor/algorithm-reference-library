# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
#
# Synthesise and Image interferometer data
"""Parameter name meanings:

- p: The uvw coordinates [*,3] (m)
- v: The Visibility values [*] (Jy)
- field_of_view: Width of the field of view to be synthetised, as directional
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
- `ceil(field_of_vietest_synthesis.pyw * lam)` gives the image size `N` in pixels
- `lam * coordinates2(N)` yields the `u,v` grid coordinate system
- `field_of_view * coordinates2(N)` yields the `l,m` image coordinate system
   (radians, roughly)
"""

from __future__ import division

import numpy
from arl.convolutional_gridding import *

def convolutional_grid(gcf, a, uv, v):
    """Grid after convolving with gcf

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param a: Grid to add to
    :param uv: UVW positions
    :param v: Visibility values
    :param gcf: Oversampled convolution kernel
    """
    
    Qpx, _, gh, gw = gcf.shape
    coords = _frac_coords(a.shape, Qpx, uv)
    for v, x, xf, y, yf in zip(v, *coords):
        a[y - gh // 2: y + (gh + 1) // 2,
        x - gw // 2: x + (gw + 1) // 2] += gcf[yf, xf] * v


def convolutional_degrid(gcf, a, uv):
    """Convolutional degridding

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param gcf: Oversampled convolution kernel
    :param a:   The uv plane to de-grid from
    :param p:   The coordinates to degrid at.
    :returns: Array of visibilities.
    """
    Qpx, _, gh, gw = gcf.shape
    coords = _frac_coords(a.shape, Qpx, uv)
    vis = [
        numpy.sum(a[y - gh // 2: y + (gh + 1) // 2,
                  x - gw // 2: x + (gw + 1) // 2] * gcf[yf, xf])
        for x, xf, y, yf in zip(*coords)
        ]
    return numpy.array(vis)


def do_weight(field_of_view, uvmax, uv, v):
    """Re-weight visibilities

    Incorrect for MFS
    Note that as is usual, convolution kernels are not taken into account
    TODO: Replace by more general version
    """
    N = int(round(field_of_view * uvmax))
    assert N > 1
    gw = numpy.zeros([N, N])
    x, xf, y, yf = _frac_coords(gw.shape, 1, uv / uvmax)
    for i in range(len(x)):
        gw[x[i], y[i]] += 1
    v = v.copy()
    for i in range(len(x)):
        v[i] /= gw[x[i], y[i]]
    return v


def convolutional_imaging(field_of_view, uvmax, uv, v, kv):
    """Convolve and grid with user-supplied kernels

    :param field_of_view: Field of view (directional cosines))
    :param uvmax:UV grid range
    :param uv: UVWs of visibilities
    :param v: Visibility values
    :param kv: Gridding kernel
    :returns: UV grid
    """
    N = int(round(field_of_view * uvmax))
    assert N > 1
    guv = numpy.zeros([N, N], dtype=complex)
    convolutional_grid(kv, guv, uv / uvmax, v)
    return guv


def simulate_point(dist_uvw, l, m):
    """
    Simulate visibilities for unit amplitude point source at
    direction cosines (l,m) relative to the phase centre.

    This includes phase tracking to the centre of the field (hence the minus 1
    in the exponent.)

    Note that point source is delta function, therefore the
    FT relationship becomes an exponential, evaluated at
    (uvw.lmn)

    :param dist_uvw: :math:`(u,v,w)` distribution of projected baselines (in wavelengths)
    :param l: horizontal direction cosine relative to phase tracking centre
    :param m: orthogonal directon cosine relative to phase tracking centre
    """
    
    # vector direction to source
    s = numpy.array([l, m, numpy.sqrt(1 - l ** 2 - m ** 2) - 1.0])
    # complex valued Visibility data
    return numpy.exp(-2j * numpy.pi * numpy.dot(dist_uvw, s))


def visibility_shift(uvw, vis, dl, dm):
    """
    Shift visibilities by the given image-space distance. This is
    based on simple FFT laws. It will require kernels to be suitably
    shifted as well to work correctly.

    :param vis: :math:`(u,v,w)` distribution of projected baselines (in wavelengths)
    :param vis: Input visibilities
    :param dl: Horizontal shift distance as directional cosine
    :param dm: Vertical shift distance as directional cosine
    :returns: New visibilities

    """
    
    s = numpy.array([dl, dm])
    return vis * numpy.exp(-2j * numpy.pi * numpy.dot(uvw[:, 0:2], s))


def uvw_transform(uvw, T):
    """
    Transforms UVW baseline coordinates such that the image is
    transformed with the given matrix. Will require kernels to be
    suitably transformed to work correctly.

    Reference: Sault, R. J., L. Staveley-Smith, and W. N. Brouw. "An
    approach to interferometric mosaicing." Astronomy and Astrophysics
    Supplement Series 120 (1996): 375-384.

    :param uvw: :math:`(u,v,w)` distribution of projected baselines (in wavelengths)
    :param T: 2x2 matrix for image transformation
    :returns: New baseline coordinates
    """
    
    # Calculate transformation matrix (see Sault)
    Tt = numpy.linalg.inv(numpy.transpose(T))
    # Apply to uv coordinates
    uv1 = numpy.dot(uvw[:, 0:2], Tt)
    # Restack with original w values
    return numpy.hstack([uv1, uvw[:, 2:3]])
