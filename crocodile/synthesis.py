# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
#
# Synthetise and Image interferometer data
"""Parameter name meanings:
- p: The uvw coordinates [*,3] (m)
- v: The Visibility values [*] (Jy)
- T2: Theta2, the half-width of the field of view to be synthetised (radian)
- L2: Half-width of the uv-plane (unitless). Controls resolution of the images
- Qpx: Oversampling of pixels by the convolution kernels -- there are
    (Qpx x Qpx) convolution kernels per pixels to account for fractional
    pixel values.
"""

from __future__ import division

import numpy
import scipy.special
import pylru


def ceil2(x):
    """Find next greater power of 2"""
    return 1 << (x - 1).bit_length()


def ucsBounds(N):
    r"""Returns the lowest and highest coordinates of the array if we
    assume that it is centered consistently with `fftshift`. To be
    concrete, this must satisfy two properties:

    1. A grid with bounds :math:`[high,low]` has step size :math:`1/N`:

       .. math:: \frac{high-low}{N-1} = \frac{1}{N}

    2. The coordinate :math:`\lfloor N/2\rfloor` falls exactly on zero:

       .. math:: low + \left\lfloor\frac{N}{2}\right\rfloor * (high-low) = 0
    """
    if N % 2 == 0:
        return -0.5, 0.5 * (N - 2) / N
    else:
        return -0.5 * (N - 1) / N, 0.5 * (N - 1) / N


def ucsN(N):
    """Two dimensional grids of coordinates spanning -1 to 1 in each
    dimension, with

    1. a step size of 2/N and
    2. (0,0) at pixel (floor(n/2),floor(n/2))
    """
    low, high = ucsBounds(N)
    return numpy.mgrid[low:high:(N * 1j), low:high:(N * 1j)]


def uax2(N, eps=0):
    """1D array which spans -1 to 1 with 0 at position N/2"""
    return numpy.mgrid[(-1 + eps):(1.0 - eps) * (N - 2) / N:1j * N]


def aaf(a, m, c):
    """Compute the anti-aliasing function as separable product. See VLA
    Scientific Memoranda 129, 131, 132

    """
    sx, sy = map(lambda i: scipy.special.pro_ang1(m, m, c, uax2(a.shape[i], eps=1e-10))[0],
                 [0, 1])
    return numpy.outer(sx, sy)


def pxoversample(ff, N, Qpx, s):
    """Takes a farfield pattern and creates oversampled convolution
    functions from it.

    :param ff: Far field pattern
    :param N:  Image size
    :param Qpx: Factor to oversample by -- there will be Qpx x Qpx convolution functions
    :param s: Size of convolution function to extract
    :returns: List of of shape (Qpx, Qpx), with the inner list on the
              u coordinate and outer indexed by j
    """

    # Pad the far field to the required pixel size
    padff = wkernpad(ff, N * Qpx)

    # Obtain oversampled uv-grid
    af = numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.ifftshift(padff)))

    # Extract kernels
    res = [[wextract(af, x, y, Qpx, s) for x in range(Qpx)] for y in range(Qpx)]
    return numpy.array(res)


def wkernff(N, theta, w):
    """W beam (i.e., w effect in the far-field).

    :param theta:
    :param N:  Size of the field in pixels
    :param w: The w value

    :returns: N x N array with the far field
    """

    m, l = ucsN(N) * theta
    ph = w * (1 - numpy.sqrt(1 - l ** 2 - m ** 2))
    cp = numpy.exp(2j * numpy.pi * ph)
    return cp


def wkernpad(ff, N):
    """Pad a far field Image with zeroes to make it the given size. In
    comparison of creating a far field of the full size right away,
    this means that we "limit" the pattern, which would be a
    convolution with a sinc pattern in the uv-grid.

    :param ff: The input far field. Should be smaller than NxN.
    :param N:  The desired far field size
    """

    N0 = ff.shape[0]
    if N == N0: return ff
    assert N > N0
    return numpy.pad(ff,
                     pad_width=[((N - N0 + 1) // 2, (N - N0) // 2), ((N - N0 + 1) // 2, (N - N0) // 2)],
                     mode='constant',
                     constant_values=(0.0,))


def wkernsinc(N, theta, over=1):
    """Pattern for sub-pixel averaging the Image on the uv-plane. This
    conforms to convolving the uv-grid with a one-pixel box function.

    :param over:
    :param N:     Size of the field in pixels
    :param theta: Width of far-field in radian
    """

    # Note that theta actually cancels out here - the pattern only
    # depends on the pixel size we do the FFT with.
    lambda_pix = 1 / theta  # Size of a pixel in the uv-plane
    m, l = ucsN(N) * theta / 2
    return numpy.sinc(lambda_pix * l * over) * numpy.sinc(lambda_pix * m * over)


def wextract(a, x, y, Qpx, s):
    """Extract the (xth,yth) w-kernel from the oversampled parent

    The kernel is reversed in order to make the suitable for
    correcting the fractional coordinates
    """
    a = a[y::Qpx, x::Qpx]  # view every Qpx-th pixel
    a = exmid2(a, s)  # extract middle
    a = a[::-1, ::-1]  # reverse
    return Qpx * Qpx * a  # normalise


def wkernaf2(N, lam, w):
    """

    :param N:
    :param lam:
    :param w:
    :return:
    """
    duv = T2 / N
    vs, us = ucsN(N) * lam
    vs0 = vs - duv;
    vs1 = vs + duv
    us0 = vs - duv;
    us1 = us + duv
    from scipy.special import erf
    f = (1j - 1) * numpy.sqrt(numpy.pi / 2 / w)
    return -(erf(vs0) - erf(vs1)) * (erf(us0) - erf(us1)) / 4


def wkernaf(N, theta, w, s,
            Qpx):
    """
    The middle s pixels of W convolution kernel. (W-KERNel-Aperture-Function)

    :param N:
    :param theta:
    :param w:
    :param s:
    :param Qpx: Oversampling, pixels will be Qpx smaller in aperture
      plane than required to minimially sample theta.

    :return: (Qpx,Qpx) shaped list of convolution kernels
    """
    wff = wkernff(N, theta, w)
    return pxoversample(wff, N, Qpx, s)


def kinvert(a):
    """
    Pseudo-Invert a kernel: element-wise inversion (see RauThesis2010:Eq4.6)
    """
    return numpy.conj(a) / (numpy.abs(a) ** 2)


def sample(a, p):
    """Take samples from array a

    This should use numpy.around as grid does.
    """
    x = ((1 + p[:, 0]) * a.shape[0] / 2).astype(int)
    y = ((1 + p[:, 1]) * a.shape[1] / 2).astype(int)
    return a[x, y]


def grid(a, p, v):
    """Grid visibilies (v) at positions (p) into (a) without convolution

    The zeroth frequency is at N/2 where N is the length on side of
    the grid.
    """
    assert numpy.max(p) < 0.5

    x = numpy.around((0.5 + p[:, 0]) * a.shape[0]).astype(int)
    y = numpy.around((0.5 + p[:, 1]) * a.shape[1]).astype(int)
    for i in range(len(x)):
        a[x[i], y[i]] += v[i]
    return a


def degrid(a, p):
    """DeGrid visibilities (v) at positions (p) from (a) without convolution

    The zeroth frequency is at N/2 where N is the length on side of
    the grid.
    :param a:   The uv plane to de-grid from
    :param p:   The coordinates to degrid at (in fraction of grid)

    :returns: Array of visibilities.
    """
    assert numpy.max(p) < 0.5

    x = numpy.around((0.5 + p[:, 0]) * a.shape[0]).astype(int)
    y = numpy.around((0.5 + p[:, 1]) * a.shape[1]).astype(int)
    v = []
    for i in range(len(x)):
        v.append(a[x[i], y[i]])
    return numpy.array(v)


def convgridone(a, p, gcf, v):
    """Convolve and grid one Visibility sample"""
    x, xf, y, yf = convcoordsone(a, p, len(gcf))
    sx, sy = gcf[0][0].shape[0] // 2, gcf[0][0].shape[1] // 2
    sx, sy = gcf[0][0].shape[0] // 2, gcf[0][0].shape[1] // 2
    a[int(x) - sx: int(x) + sx + 1,
    int(y) - sy: int(y) + sy + 1] += gcf[xf][yf] * v


def convdegridone(a, p, gcf):
    """Convolutional-degridding for one Visibility sample

    :param a:   The uv plane to de-grid from
    :param p:   The coordinates to degrid at.
    :param gcf: List of convolution kernels

    :returns: Visibility valu
    """
    x, xf, y, yf = convcoordsone(a, p, len(gcf))
    sx, sy = gcf[0][0].shape[0] // 2, gcf[0][0].shape[1] // 2
    return (a[x - sx: x + sx + 1, y - sy: y + sy + 1] * gcf[xf][yf]).sum()


def fraccoord(N, p, Qpx):
    """Compute whole and fractional parts of coordinates, rounded to Qpx-th fraction of pixel size

    :param N: Number of pixels in total
    :param p: coordinates in range [-.5,.5[
    :param Qpx: Fractional values to round to
    """
    x = (.5 + p) * N
    flx = numpy.floor(x + 0.5 / Qpx)
    fracx = numpy.around(((x - flx) * Qpx))
    return flx.astype(int), fracx.astype(int)


def convcoords(a, p, Qpx):
    """Compute grid coordinates and fractional values for convolutional
    gridding

    The fractional values are rounded to nearest 1/Qpx pixel value at
    fractional values greater than (Qpx-0.5)/Qpx are roundeded to next
    integer index
    """
    (x, xf), (y, yf) = [fraccoord(a.shape[i], p[:, i], Qpx) for i in [0, 1]]
    return x, xf, y, yf


def convcoordsone(a, p, Qpx):
    """Compute grid coordinates and fractional values for convolutional
    gridding for a single Visibility

    The fractional values are rounded to nearest 1/Qpx pixel value at
    fractional values greater than (Qpx-0.5)/Qpx are roundeded to next
    integer index
    """
    (x, xf), (y, yf) = [fraccoord(a.shape[i], p[i], Qpx) for i in [0, 1]]
    return x, xf, y, yf


def convgrid(a, p, v, gcf):
    """Grid after convolving with gcf, taking into account fractional uv
    cordinate values

    :param a: Grid to add to
    :param p: UVW positions
    :param v: Visibility values
    :param gcf: List  (shape Qpx, Qpx) of convolution kernels of
    """
    for i in range(len(p)):
        convgridone(a, p[i], gcf, v[i])

    return a


def convdegrid(a, p, gcf):
    """Convolutional-degridding

    :param a:   The uv plane to de-grid from
    :param p:   The coordinates to degrid at.
    :param gcf: List of convolution kernels

    :returns: Array of visibilities.
    """
    x, xf, y, yf = convcoords(a, p, len(gcf))
    v = []
    sx, sy = gcf[0][0].shape[0] // 2, gcf[0][0].shape[1] // 2
    for i in range(len(x)):
        pi = (x[i], y[i])
        v.append((a[pi[0] - sx: pi[0] + sx + 1, pi[1] - sy: pi[1] + sy + 1] * gcf[xf[i]][yf[i]]).sum())
    return numpy.array(v)


def exmid2(a, s):
    """Extract a section from middle of a map, suitable for zero
    frequencies at N/2. For even dimensions, this is the reverse
    operation to wkernpad.

    param: a: array from which extract is taken
    param: s: size of section is 2s+1
    """
    cx = a.shape[0] // 2
    cy = a.shape[1] // 2
    return a[cx - s:cx + s + 1, cy - s:cy + s + 1]


def div0(a1, a2):
    """Divide a1 by a2 except pixels where a2 is zero"""
    m = (a2 != 0)
    res = a1.copy()
    res[m] /= a2[m]
    return res


def halfinv(g):
    """Invert a hermitian-symetric two-dimensional grid.

    The hermitian symetric dimension is the second (last) index, like
    in `numpy.fft`

    :param g: The uv grid to invert. Note that the zero frequency is
    expected at pixel N/2 where N is the size of the grid on the side.

    This function is like doing `ifftshift` on the x axis but not on the
    y axis

    """
    Nx, Ny = g.shape
    huv = numpy.roll(g[:, (Ny // 2):], shift=-(Nx - 1) // 2, axis=0)
    huv = numpy.pad(huv,
                    pad_width=((0, 0), (0, 1)),
                    mode='constant',
                    constant_values=0)
    return numpy.fft.fftshift(numpy.fft.irfft2(huv))


def inv(g):
    """Invert a complex two-dimensional grid.

    :param g: The uv grid to invert. Note that the zero frequency is
    expected at pixel N/2 where N is the size of the grid on the side.

    """
    return numpy.real(numpy.fft.ifftshift(numpy.fft.ifft2(g)))


def rotv(p, l, m, v):
    """Rotate visibilities to direction (l,m)"""
    s = numpy.array([l, m, numpy.sqrt(1 - l ** 2 - m ** 2)])
    return (v * numpy.exp(2j * numpy.pi * numpy.dot(p, s)))


def rotw(p, v):
    """Rotate visibilities to zero w plane"""
    return rotv(p, 0, 0, v)


def posvv(p, v):
    """Move all visibilities to the positive v half-plane"""
    s = p[:, 1] < 0
    p[s] *= -1.0
    v[s] = numpy.conjugate(v[s])
    return p, v


def sortw(p, v):
    """Sort on the w value. (p) are uvw coordinates and (v) are the
    Visibility values
    """
    zs = numpy.argsort(p[:, 2])
    if v is not None:
        return p[zs], v[zs]
    else:
        return p[zs]


def doweight(theta, lam, p, v):
    """Re-weight visibilities

    Note that as is usual, convolution kernels are not taken into account
    """
    N = int(round(theta * lam))
    assert N > 1
    gw = numpy.zeros([N, N])
    x, xf, y, yf = convcoords(gw, p / lam, 1)
    for i in range(len(x)):
        gw[x[i], y[i]] += 1
    v = v.copy()
    for i in range(len(x)):
        v[i] /= gw[x[i], y[i]]
    return v


def simpleimg(theta, lam, p, v):
    """Trivial function for imaging which does no convolution but simply
    puts the visibilities into a grid cell i.e. boxcar gridding"""
    N = int(round(theta * lam))
    assert N > 1
    guv = numpy.zeros([N, N], dtype=complex)
    grid(guv, p / lam, v)
    return guv


def simplepredict(guv, theta, lam, p):
    """Trivial function for degridding which does no convolution but simply
    extracts the visibilities from a grid cell i.e. boxcar degridding"""
    N = int(round(theta * lam))
    assert N > 1
    v = degrid(guv, p / lam)
    p[:, 2] *= -1
    return p, v


def convimg(theta, lam, p, v, kv):
    """Convolve and grid with user-supplied kernels"""
    N = int(round(theta * lam))
    assert N > 1
    guv = numpy.zeros([N, N], dtype=complex)
    convgrid(guv, p / lam, v, kv)
    return guv


def wslicimg(theta, lam, p, v,
             wstep=2000,
             Qpx=4,
             NpixFF=256,
             NpixKern=15):
    """Basic w-projection by w-sort and slicing in w

    :param theta:
    :param lam:
    :param Qpx:
    :param p: UVWs of visiblities
    :param v: Visibility values
    :param wstep: The step between w-slices (pixels). W kernels are re-computed
      for each slice, for the mean of the w-coordinates of all
      visibilities falling into the slides.
    :param NpixFF: Size of the far-field for computing the
      w-kernel. See doc/wkernel.
    :param NpixKern: Size of the extracted convolution
      kernels. Currently kernels are the same size for all w-values.
    """
    N = int(round(theta * lam))
    assert N > 1
    guv = numpy.zeros([N, N], dtype=complex)
    p, v = sortw(p, v)
    nv = len(v)
    ii = range(0, nv, wstep)
    ir = list(zip(ii[:-1], ii[1:])) + [(ii[-1], nv)]
    for ilow, ihigh in ir:
        w = p[ilow:ihigh, 2].mean()
        wg = wkernaf(NpixFF, theta, w, NpixKern, Qpx)
        wg = list(map(lambda x: list(map(numpy.conj, x)), wg))
        convgrid(guv, p[ilow:ihigh] / lam, v[ilow:ihigh], wg)
    return guv


def wslicfwd(guv,
             theta, lam, p,
             wstep=2000,
             Qpx=4,
             NpixFF=256,
             NpixKern=15):
    """Predict visibilities using w-slices

    :param Qpx:
    :param guv: Input uv plane to de-grid from
    :param theta: Field of view in radians
    :param lam: Observing wavelength (m)
    :param p: UVWs of visiblities
    :param wstep: The step between w-slices (pixels). W kernels are re-computed
      for each slice, for the mean of the w-coordinates of all
      visibilities falling into the slices.
    :param NpixFF: Size of the far-field for computing the
      w-kernel. See doc/wkernel.
    :param NpixKern: Size of the extracted convolution
      kernels. Currently kernels are the same size for all w-values.

    :returns w-sorted uvw, w-sorted degridded visibilities
    """
    # Calculate number of pixels in the Image
    N = int(round(theta * lam))
    assert N > 1
    # Sort the u,v,w coordinates
    p = sortw(p, None)
    nv = len(p)
    ii = range(0, nv, wstep)
    # Make a tuple containing begin and end indices
    res = []
    ii = range(0, nv, wstep)
    ir = list(zip(ii[:-1], ii[1:])) + [(ii[-1], nv)]
    # Loop over all visibilities in block of wstep
    # average w to calculate the gridding kernel
    for ilow, ihigh in ir:
        w = p[ilow:ihigh, 2].mean()
        wg = wkernaf(NpixFF, theta, w, NpixKern, Qpx)
        res.append(convdegrid(guv, p[ilow:ihigh] / lam, wg))
    v = numpy.concatenate(res)
    p[:, 2] *= -1
    return p, v


def wcacheimg(theta, lam, p, v,
              wcache=None,
              wstep=2000,
              cachesize=10000,
              Qpx=4,
              NpixFF=256,
              NpixKern=15):
    """Basic w-projection by caching convolution functions in w

    The cache can be constructed externally and passed in:

    wcache=pylru.FunctionCacheManager(lambda iw: wkernaf(NpixFF, theta, iw*wstep, NpixKern, Qpx), cachesize)

    :param theta:
    :param lam:
    :param wcache:
    :param cachesize:
    :param Qpx:
    :param p: UVWs of visiblities
    :param v: Visibility values
    :param wstep: The binning of w values. W kernels are computed
        for each bin, and then cached
    :param NpixFF: Size of the far-field for computing the
      w-kernel. See doc/wkernel.
    :param NpixKern: Size of the extracted convolution
      kernels. Currently kernels are the same size for all w-values.
    """
    N = int(round(theta * lam))
    assert N > 1
    p, v = sortw(p, v)
    guv = numpy.zeros([N, N], dtype=complex)
    if wcache is None:
        print("Making w-kernel cache of %d kernels" % (cachesize))
        wcache = pylru.FunctionCacheManager(lambda iw: wkernaf(NpixFF, theta, iw * wstep, NpixKern, Qpx), cachesize)
    for iv in range(len(v)):
        iw = int(round(p[iv, 2] / wstep))
        convgridone(guv, p[iv] / lam, wcache(iw), v[iv])
    return guv


def wcachefwd(guv,
              theta, lam, p,
              wcache=None,
              wstep=2000,
              cachesize=10000,
              Qpx=4,
              NpixFF=256,
              NpixKern=15):
    """Predict visibilities using w-kernel cache

    The cache can be constructed externally and passed in:

    wcache=pylru.FunctionCacheManager(lambda iw: wkernaf(NpixFF, theta, iw*wstep, NpixKern, Qpx), cachesize)

    :param wcache:
    :param cachesize:
    :param Qpx:
    :param NpixFF:
    :param guv: Input uv plane to de-grid from
    :param theta: Field of view in radians
    :param lam: Observing wavelength (m)
    :param p: UVWs of visiblities
    :param wstep: The binning of w values. W kernels are computed
        for each bin, and then cached
    :param NpixKern: Size of the extracted convolution
      kernels. Currently kernels are the same size for all w-values.

    :returns degridded visibilities
    """
    # Calculate number of pixels in the Image
    N = int(round(theta * lam))
    assert N > 1
    nv = p.shape[0]
    if wcache is None:
        print("Making w-kernel cache of %d kernels" % (cachesize))
        wcache = pylru.FunctionCacheManager(lambda iw: wkernaf(NpixFF, theta, iw * wstep, NpixKern, Qpx), cachesize)
    v = numpy.zeros(nv, dtype='complex')
    for iv in range(nv):
        iw = int(round(p[iv, 2] / wstep))
        v[iv] = convdegridone(guv, numpy.array(p[iv] / lam), wcache(iw))
    p[:, 2] *= -1
    return p, v


def doimg(theta, lam, p, v, imgfn):
    """Do imaging with imaging function (imgfn)

    Note: No longer move all visibilities to positive v -- need to
    check the convolution kernels

    :param theta:
    :param lam:
    :param p: UVWs of visibilities (m)
    :param v: Visibilities to be imaged
    :param imgfn: imaging function e.g. wslicimg or w-slices

    :returns dirty Image, psf
    """
    # add the conjugate points
    p = numpy.vstack([p, p * -1])
    v = numpy.hstack([v, numpy.conj(v)])
    wt = doweight(theta, lam, p, numpy.ones(len(p)))
    cdrt = imgfn(theta, lam, p, wt * v)
    drt = numpy.real(numpy.fft.ifftshift(numpy.fft.ifft2(numpy.fft.ifftshift(cdrt))))
    c = imgfn(theta, lam, p, wt)
    psf = numpy.real(numpy.fft.ifftshift(numpy.fft.ifft2(numpy.fft.ifftshift(c))))
    pmax = psf.max()
    assert pmax > 0.0
    return drt / pmax, psf / pmax, pmax


def dopredict(theta, lam, p, modelimage, predfn):
    """
    Predict visibilities for a model Image at the phase centre using the
    specified degridding function.

    :param theta:
    :param lam:
    :param p: UVWs of visiblities (m)
    :param modelimage: model Image as numpy.array (phase center at Nx/2,Ny/2)
    :param predfn: prediction function e.g. wslicfwd

    :returns predicted visibilities
    """
    ximage = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(modelimage.astype(complex))))
    return predfn(ximage, theta, lam, p)
