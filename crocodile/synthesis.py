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
- `ceil(field_of_view * lam)` gives the image size `N` in pixels
- `lam * coordinates2(N)` yields the `u,v` grid coordinate system
- `field_of_view * coordinates2(N)` yields the `l,m` image coordinate system
   (radians, roughly)
"""

from __future__ import division

import numpy
import pylru
import scipy.special


def ceil2(x):
    """Find next greater power of 2

    NOT USED
    """
    return 1 << (x - 1).bit_length()


def coordinateBounds(N):
    r"""
    Returns lowest and highest coordinates of an image/grid given:

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
    """1D array which spans [-.5,.5[ with 0 at position N/2"""
    low, high = coordinateBounds(N)
    return numpy.mgrid[low:high:(N * 1j)]

def coordinates2(N):
    """Two dimensional grids of coordinates spanning -1 to 1 in each
    dimension, with

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

    We do not want to make assumptions about the source grid's symetry
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

    # 2D Prolate spheroidal angular function is seperable
    sy, sx = [ scipy.special.pro_ang1(m, m, c, coordinates(N))[0]
               for N in shape ]
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


def invert_kernel(a):
    """
    Pseudo-Invert a kernel: element-wise inversion (see RauThesis2010:Eq4.6)

    NOT USED
    """
    return numpy.conj(a) / (numpy.abs(a) ** 2)


def grid(a, p, v):
    """Grid visibilities (v) at positions (p) into (a) without convolution

    :param a:   The uv plane to grid to (updated in-place!)
    :param p:   The coordinates to grid to (in fraction [-.5,.5[ of grid)
    :param v:   Visibilities to grid
    """
    assert numpy.max(p) < 0.5

    N = a.shape[0]
    xy = N//2 + numpy.floor(0.5 + N * p[:,0:2]).astype(int)
    for (x, y), v in zip(xy, v):
        a[y, x] += v


def degrid(a, p):
    """DeGrid visibilities (v) at positions (p) from (a) without convolution

    :param a:   The uv plane to de-grid from
    :param p:   The coordinates to degrid at (in fraction of grid)
    :returns: Array of visibilities.
    """
    assert numpy.max(p) < 0.5

    N = a.shape[0]
    xy = N//2 + numpy.floor(0.5 + p[:,0:2] * N).astype(int)
    v = [ a[y,x] for x,y in xy ]
    return numpy.array(v)


def frac_coord(N, Qpx, p):
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
    x = N//2 + p * N
    flx = numpy.floor(x + 0.5 / Qpx)
    fracx = numpy.around((x - flx) * Qpx)
    return flx.astype(int), fracx.astype(int)


def frac_coords(shape, Qpx, p):
    """Compute grid coordinates and fractional values for convolutional
    gridding

    :param shape: (height,width) grid shape
    :param Qpx: Oversampling factor
    :param p: array of (x,y) coordinates in range [-.5,.5[
    """
    h, w = shape # NB order (height,width) to match numpy!
    x, xf = frac_coord(w, Qpx, p[:,0])
    y, yf = frac_coord(h, Qpx, p[:,1])
    return x,xf, y,yf


def convgrid(gcf, a, p, v):
    """Grid after convolving with gcf

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param a: Grid to add to
    :param p: UVW positions
    :param v: Visibility values
    :param gcf: Oversampled convolution kernel
    """

    Qpx, _, gh, gw = gcf.shape
    coords = frac_coords(a.shape, Qpx, p)
    for v, x,xf, y,yf in zip(v, *coords):
        a[y-gh//2 : y+(gh+1)//2,
          x-gw//2 : x+(gw+1)//2] += gcf[yf,xf] * v


def convdegrid(gcf, a, p):
    """Convolutional degridding

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param gcf: Oversampled convolution kernel
    :param a:   The uv plane to de-grid from
    :param p:   The coordinates to degrid at.
    :returns: Array of visibilities.
    """
    Qpx, _, gh, gw = gcf.shape
    coords = frac_coords(a.shape, Qpx, p)
    vis = [
        numpy.sum(a[y-gh//2 : y+(gh+1)//2,
                    x-gw//2 : x+(gw+1)//2] * gcf[yf,xf])
        for x,xf, y,yf in zip(*coords)
    ]
    return numpy.array(vis)


def sort_vis_w(p, v=None):
    """Sort visibilities on the w value.
    :param p: uvw coordinates
    :param v: Visibility values (optional)
    """
    zs = numpy.argsort(p[:, 2])
    if v is not None:
        return p[zs], v[zs]
    else:
        return p[zs]


def slice_vis(step, p, v=None):
    """ Slice visibilities into a number of chunks.

    :param step: Maximum chunk size
    :param p: uvw coordinates
    :param v: Visibility values (optional)
    :returns: List of visibility chunk (pairs)
    """
    nv = len(p)
    ii = range(0, nv, step)
    if v is None:
        return [ p[i:i+step] for i in ii ]
    else:
        return [ (p[i:i+step], v[i:i+step]) for i in ii ]


def doweight(field_of_view, lam, p, v):
    """Re-weight visibilities

    Note that as is usual, convolution kernels are not taken into account
    """
    N = int(round(field_of_view * lam))
    assert N > 1
    gw = numpy.zeros([N, N])
    x, xf, y, yf = frac_coords(gw.shape, 1, p / lam)
    for i in range(len(x)):
        gw[x[i], y[i]] += 1
    v = v.copy()
    for i in range(len(x)):
        v[i] /= gw[x[i], y[i]]
    return v


def simple_imaging(field_of_view, lam, p, v):
    """Trivial function for imaging

    Does no convolution but simply puts the visibilities into a grid cell i.e. boxcar gridding"""
    N = int(round(field_of_view * lam))
    assert N > 1
    guv = numpy.zeros([N, N], dtype=complex)
    grid(guv, p / lam, v)
    return guv


def simple_predict(guv, field_of_view, lam, p):
    """Trivial function for degridding

    Does no convolution but simply extracts the visibilities from a grid cell i.e. boxcar degridding

    :param field_of_view: Field of view (directional cosines)
    :param uvmax:Maximum uv represented in the grid
    :param p: UVWs of visibilities
    :param v: Visibility values
    :param kv: gridding kernel
    :returns: p, v
    """
    N = int(round(field_of_view * lam))
    assert N > 1
    v = degrid(guv, p / lam)
    return p, v


def conv_imaging(field_of_view, lam, p, v, kv):
    """Convolve and grid with user-supplied kernels

    :param field_of_view: Field of view (directional cosines))
    :param uvmax:UV grid range
    :param p: UVWs of visibilities
    :param v: Visibility values
    :param kv: Gridding kernel
    :returns: UV grid
    """
    N = int(round(field_of_view * lam))
    assert N > 1
    guv = numpy.zeros([N, N], dtype=complex)
    convgrid(kv, guv, p / lam, v)
    return guv


def w_slice_imaging(field_of_view, lam, p, v,
                    wstep=2000,
                    kernel_fn=w_kernel,
                    **kwargs):
    """Basic w-projection imaging using slices

    Sorts visibility by w value and splits into equally sized slices.
    W-value used for kernels is mean w per slice. Uses the same size
    for all kernels irrespective of w.

    :param field_of_view: Field of view (directional cosines)
    :param uvmax:UV grid range (wavelenghts)
    :param p: UVWs of visibilities
    :param v: Visibility values
    :param wstep: Size of w-slices
    :param kernel_fn: Function for generating the kernels. Parameters
      `(field_of_view, w, **kwargs)`. Default `w_kernel`.
    :returns: UV grid
    """
    N = int(round(field_of_view * lam))
    assert N > 1
    slices = slice_vis(wstep, *sort_vis_w(p, v))
    guv = numpy.zeros([N, N], dtype=complex)
    for ps, vs in slices:
        w = numpy.mean(ps[:, 2])
        wg = numpy.conj(kernel_fn(field_of_view, w, **kwargs))
        convgrid(wg, guv, ps / lam, vs)
    return guv


def w_slice_predict(field_of_view, lam, p, guv,
                    wstep=2000,
                    kernel_fn=w_kernel,
                    **kwargs):
    """Basic w-projection predict using w-slices

    Sorts visibility by w value and splits into equally sized slices.
    W-value used for kernels is mean w per slice. Uses the same size
    for all kernels irrespective of w.

    :param field_of_view: Field of view (directional cosines)
    :param uvmax:UV grid range (wavelenghts)
    :param p: UVWs of visiblities
    :param guv: Input uv grid to de-grid from
    :param wstep: Size of w-slices
    :param kernel_fn: Function for generating the kernels. Parameters
      `(field_of_view, w, **kwargs)`. Default `w_kernel`.
    :returns: Visibilities, same order as p
    """
    # Calculate number of pixels in the Image
    N = int(round(field_of_view * lam))
    assert N > 1
    # Sort the u,v,w coordinates. We cheat a little and also pass
    # visibility indices so we can easily undo the sort later.
    nv = len(p)
    slices = slice_vis(wstep, *sort_vis_w(p, numpy.arange(nv)))
    v = numpy.ndarray(nv, dtype=complex)
    for ps, ixs in slices:
        w = numpy.mean(ps[:, 2])
        wg = kernel_fn(field_of_view, w, **kwargs)
        v[ixs] = convdegrid(wg, guv, ps / lam)
    return v


def w_conj_kernel_fn(kernel_fn):
    """Wrap a kernel function for which we know that

       kernel_fn(w) = conj(kernel_fn(-w))

    Such that we only evaluate the function for positive w. This is
    benificial when the underlying kernel function does caching, as it
    improves the cache hit rate.

    :param kernel_fn: Kernel function to wrap
    :returns: Wrapped kernel function
    """

    def fn(field_of_view, w, **kw):
        if w < 0:
            return numpy.conj(kernel_fn(field_of_view, -w, **kw))
        return kernel_fn(field_of_view, w, **kw)
    return fn


def w_cache_imaging(field_of_view, lam, p, v,
                    wstep=2000,
                    kernel_cache=None,
                    kernel_fn=w_kernel,
                    **kwargs):
    """Basic w-projection by caching convolution arl in w

    A simple cache can be constructed externally and passed in:

      kernel_cache = pylru.FunctionCacheManager(w_kernel, cachesize)

    If applicable, consider wrapping in `w_conj_kernel_fn` to improve
    effectiveness further.

    :param field_of_view: Field of view (directional cosines)
    :param uvmax:UV grid range (wavelenghts)
    :param p: UVWs of visibilities (wavelengths)
    :param v: Visibilites to be imaged
    :param wstep: Size of w-bins (wavelengths)
    :param kernel_cache: Kernel cache. If not passed, we fall back
       to `kernel_fn`.
    :param kernel_fn: Function for generating the kernels. Parameters
       `(field_of_view, w, **kwargs)`. Default `w_kernel`.
    :returns: UV grid

    """

    # Construct default cache, if needed. As visibilities are
    # traversed in w-order it only needs to hold the last w-kernel.
    if kernel_cache is None:
        kernel_cache = pylru.FunctionCacheManager(kernel_fn, 1)
    # Bin w values, then run slice imager with slice size of 1
    def kernel_binner(field_of_view, w, **kw):
        wbin = wstep * numpy.round(w / wstep)
        return kernel_cache(field_of_view, wbin, **kw)
    return w_slice_imaging(field_of_view, lam, p, v, 1, kernel_binner, **kwargs)


def w_cache_predict(field_of_view, lam, p, guv,
                    wstep=2000,
                    kernel_cache=None,
                    kernel_fn=w_kernel,
                    **kwargs):
    """Predict visibilities using w-kernel cache

    :param field_of_view: Field of view (directional cosines)
    :param uvmax:UV grid range (wavelenghts)
    :param p: UVWs of visibilities  (wavelengths)
    :param guv: Input uv grid to de-grid from
    :param wstep: Size of w-bins (wavelengths)
    :param kernel_cache: Kernel cache. If not passed, we fall back
       to `kernel_fn`. See `w_cache_imaging` for details.
    :param kernel_fn: Function for generating the kernels. Parameters
       `(field_of_view, w, **kwargs)`. Default `w_kernel`.
    :returns: degridded visibilities
    """

    if kernel_cache is None:
        kernel_cache = pylru.FunctionCacheManager(kernel_fn, 1)
    def kernel_binner(field_of_view, w, **kw):
        wbin = wstep * numpy.round(w / wstep)
        return kernel_cache(field_of_view, wbin, **kw)
    return w_slice_predict(field_of_view, lam, p, guv, 1, kernel_binner, **kwargs)


def do_imaging(field_of_view, lam, p, v, imgfn, **kwargs):
    """Do imaging with imaging function (imgfn)

    :param field_of_view: Field of view (directional cosines)
    :param uvmax:UV grid range (wavelenghts)
    :param p: UVWs of visibilities (wavelengths)
    :param v: Visibilities to be imaged
    :param imgfn: imaging function e.g. `simple_imaging`, `conv_imaging`,
      `w_slice_imaging` or `w_cache_imaging`. All keyword parameters
      are passed on to the imaging function.
    :returns: dirty Image, psf
    """
    # Add the conjugate points
    p = numpy.vstack([p, p * -1])
    v = numpy.hstack([v, numpy.conj(v)])
    # Determine weights
    wt = doweight(field_of_view, lam, p, numpy.ones(len(p)))
    # Make image
    cdrt = imgfn(field_of_view, lam, p, wt * v, **kwargs)
    drt = numpy.real(ifft(cdrt))
    # Make point spread function
    c = imgfn(field_of_view, lam, p, wt, **kwargs)
    psf = numpy.real(ifft(c))
    # Normalise
    pmax = psf.max()
    assert pmax > 0.0
    return drt / pmax, psf / pmax, pmax


def do_predict(field_of_view, lam, p, modelimage, predfn, **kwargs):
    """Predict visibilities for a model Image at the phase centre using the
    specified degridding function.

    :param field_of_view: Field of view (directional cosines)
    :param uvmax:UV grid range (wavelenghts)
    :param p: UVWs of visiblities (wavelengths)
    :param modelimage: model image as numpy.array (phase center at Nx/2,Ny/2)
    :param predfn: prediction function e.g. `simple_predict`,
      `w_slice_predict` or `w_cache_predict`.
    :returns: predicted visibilities
    """
    ximage = fft(modelimage.astype(complex))
    return predfn(field_of_view, lam, p, ximage, **kwargs)
