# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
#
# Synthesise and Image interferometer data
"""Convolutional gridding support functions

All functions that involve convolutional gridding are kept here.
"""

from __future__ import division

import logging

import scipy.special
from astropy.constants import c

from arl.data.parameters import get_parameter
from arl.fourier_transforms.fft_support import *

# from arl.core.c import gridder

log = logging.getLogger(__name__)


def coordinateBounds(npixel):
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


def coordinates(npixel: int) -> object:
    """ 1D array which spans [-.5,.5[ with 0 at position npixel/2
    
    """
    return (numpy.arange(npixel) - npixel//2) / npixel


def coordinates2(npixel: int):
    """Two dimensional grids of coordinates spanning -1 to 1 in each dimension

    1. a step size of 2/npixel and
    2. (0,0) at pixel (floor(n/2),floor(n/2))
    """
    return (numpy.mgrid[0:npixel,0:npixel] - npixel//2) / npixel


def anti_aliasing_transform(shape, oversampling=8, support=3, m=6, c=0.0):
    """
    Compute the prolate spheroidal anti-aliasing function
    
    Return the 2D grid correction function, and the convolving kernel
    
    This is not sufficiently accurate: use anti_aliasing_calculate instead.
    
    See VLA Scientific Memoranda 129, 131, 132
    :param shape: (height, width) pair
    """
    # 2D Prolate spheroidal angular function is separable
    sy, sx = [scipy.special.pro_ang1(m, m, c, 2.0 * coordinates(npixel))[0] for npixel in shape]
    sx[0] = 0.0
    sy[0] = 0.0
    gcf = numpy.outer(sy, sx)
    
    # Calculate the gridding kernel by Fourier transform of the gcf
    kernel = kernel_oversample(gcf, shape[0], oversampling, oversampling)
    kernel = kernel / kernel.max()
    gcf[gcf > 0.0] = gcf.max() / gcf[gcf > 0.0]
    return gcf, kernel


def anti_aliasing_calculate(shape, oversampling=8, support=3):
    """
    Compute the prolate spheroidal anti-aliasing function
    
    The kernel is to be used in gridding visibility data onto a grid on for degridding from a grid.
    The gridding correction function (gcf) is used to correct the image for decorrelation due to
    gridding.
    
    Return the 2D grid correction function (gcf), and the convolving kernel (kernel

    See VLA Scientific Memoranda 129, 131, 132
    :param shape: (height, width) pair
    :param oversampling: Number of sub-samples per grid pixel
    :param support: Support of kernel (in pixels) width is 2*support+2
    """
    
    # 2D Prolate spheroidal angular function is separable
    ny, nx = shape
    nu = numpy.abs(2.0 * coordinates(nx))
    gcf1d, _ = grdsf(nu)
    gcf = numpy.outer(gcf1d, gcf1d)
    gcf[gcf > 0.0] = gcf.max() / gcf[gcf > 0.0]
    
    s1d = 2 * support + 2
    nu = numpy.arange(-support, +support, 1.0 / oversampling)
    kernel1d = grdsf(nu / support)[1]
    l1d = len(kernel1d)
    # Rearrange to get the convolution function isolated by (yf, xf). For this convolution function
    # the result is heavily redundant but it does fit well into the general framework
    kernel4d = numpy.zeros((oversampling, oversampling, s1d, s1d))
    for yf in range(oversampling):
        my = range(yf, l1d, oversampling)[::-1]
        for xf in range(oversampling):
            mx = range(xf, l1d, oversampling)[::-1]
            kernel4d[yf, xf, 2:, 2:] = numpy.outer(kernel1d[my], kernel1d[mx])
    return gcf, (kernel4d / kernel4d.max()).astype('complex')


def anti_aliasing_box(shape, oversampling=1, support=1):
    """ The grid correction for a box car gridding

    :param shape: (height, width) pair
    :param oversampling: Number of sub-samples per grid pixel
    :param support: Support of kernel (in pixels) width is 2*support+2
    """
    
    # 2D Prolate spheroidal angular function is separable
    ny, nx = shape
    nu = numpy.abs(2.0 * coordinates(nx))
    gcf1d = correct_finite_oversampling(nu, oversampling=1)
    gcf = numpy.outer(gcf1d, gcf1d)
    gcf[gcf > 0.0] = gcf.max() / gcf[gcf > 0.0]
    
    return gcf, None


def grdsf(nu):
    """"Calculate PSWF using an old SDE routine re-written in Python

    Find Spheroidal function with M = 6, alpha = 1 using the rational
    approximations discussed by Fred Schwab in 'Indirect Imaging'.
    This routine was checked against Fred's SPHFN routine, and agreed
    to about the 7th significant digit.
    The gridding function is (1-NU**2)*GRDSF(NU) where NU is the distance
    to the edge. The grid correction function is just 1/GRDSF(NU) where NU
    is now the distance to the edge of the image.
    """
    p = numpy.array([[8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
                     [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2]])
    q = numpy.array([[1.0000000e0, 8.212018e-1, 2.078043e-1],
                     [1.0000000e0, 9.599102e-1, 2.918724e-1]])
    
    _, np = p.shape
    _, nq = q.shape
    
    nu = numpy.abs(nu)
    
    nuend = numpy.zeros_like(nu)
    part = numpy.zeros(len(nu), dtype='int')
    part[(nu >= 0.0) & (nu < 0.75)] = 0
    part[(nu > 0.75) & (nu < 1.0)] = 1
    nuend[(nu >= 0.0) & (nu <= 0.75)] = 0.75
    nuend[(nu > 0.75) & (nu < 1.0)] = 1.0
    
    delnusq = nu ** 2 - nuend ** 2
    
    top = p[part, 0]
    for k in range(1, np):
        top += p[part, k] * numpy.power(delnusq, k)
    
    bot = q[part, 0]
    for k in range(1, nq):
        bot += q[part, k] * numpy.power(delnusq, k)
    
    grdsf = numpy.zeros_like(nu)
    ok = (bot > 0.0)
    grdsf[ok] = top[ok] / bot[ok]
    ok = numpy.abs(nu > 1.0)
    grdsf[ok] = 0.0
    
    # Return the gridding function and the grid correction function
    return grdsf, (1 - nu ** 2) * grdsf


def correct_finite_oversampling(nu, oversampling=8):
    """Correct for the loss incurred by finite oversampling
    
    This is just a correction for a boxcar of width 1/oversampling. For oversampling=8, it's about 0.65% which is
    less than the accuracy we have so far.
    """
    result = numpy.ones_like(nu)
    nu_scaled = 0.5 * numpy.pi * nu / float(oversampling)
    result[nu != 0.0] = numpy.sin(nu_scaled[nu != 0.0]) / nu_scaled[nu != 0.0]
    return result


def w_beam(npixel, field_of_view, w):
    """ W beam, the fresnel diffraction pattern arising from non-coplanar baselines
    
    Note that we also include the anti-aliasing kernel since we will need to for
    small values of w.

    :param npixel: Size of the grid in pixels
    :param field_of_view: Field of view
    :param w: Baseline distance to the projection plane
    :returns: npixel x npixel array with the far field
    """
    
    m, l = coordinates2(npixel) * field_of_view
    r2 = l ** 2 + m ** 2
    n2 = 1.0 - r2
    ph = numpy.zeros_like(n2)
    ph[n2 < 1.0] = w * (1 - numpy.sqrt(1.0 - r2[n2 < 1.0]))
    cp = numpy.zeros_like(n2, dtype='complex')
    cp[n2 < 1.0] = numpy.exp(-2j * numpy.pi * ph[n2 < 1.0])
    return cp


def kernel_coordinates(npixel, field_of_view, dl=0, dm=0, transform_matrix=None):
    """ Returns (l,m) coordinates for generation of kernels in a far-field of the given size.

    If coordinate transformations are passed, they must be inverse to
    the transformations applied to the visibilities using
    visibility_shift/uvw_transform.

    :param field_of_view: In radians
    :param npixel: Desired far-field size
    :param dl: Pattern horizontal shift (see visibility_shift)
    :param dm: Pattern vertical shift (see visibility_shift)
    :param transformmatrix: Pattern transformation matrix (see uvw_transform)
    :returns: Pair of (m,l) coordinates
    """
    
    m, l = coordinates2(npixel) * field_of_view
    if transform_matrix is not None:
        l, m = transform_matrix[0, 0] * l + transform_matrix[1, 0] * m, transform_matrix[0, 1] * l \
               + transform_matrix[1, 1] * m
    return m + dm, l + dl


def kernel_oversample(ff, npixel, kernel_oversampling, kernelwidth):
    """ Takes a farfield pattern and creates an oversampled convolution function.

    If the far field size is smaller than npixel*kernel_oversampling, we will pad it. This
    essentially means we apply a sinc anti-aliasing kernel by default.

    :param ff: Far field pattern
    :param npixel: Image size without oversampling
    :param kernel_oversampling: Factor to oversample by -- there will be kernel_oversampling x kernel_oversampling
    convolution functions
    :param kernelwidth: Size of convolution function to extract
    :returns: Numpy array of shape [ov, ou, v, u], e.g. with sub-pixel offsets as the outer coordinates.
    """
    
    # Pad the far field to the required pixel size
    padff = pad_mid(ff, npixel * kernel_oversampling)
    
    # Obtain oversampled uv-grid
    af = ifft(padff)

    # Extract kernels
    res = [[extract_oversampled(af, x, y, kernel_oversampling, kernelwidth)
            for x in range(kernel_oversampling)]
           for y in range(kernel_oversampling)]
    return numpy.array(res)


def w_kernel(field_of_view, w, npixel_farfield, npixel_kernel, kernel_oversampling):
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
    gcf, _ = anti_aliasing_calculate((npixel_farfield, npixel_farfield), kernel_oversampling)
    wbeamarray=w_beam(npixel_farfield, field_of_view, w) / gcf
    return kernel_oversample(wbeamarray, npixel_farfield, kernel_oversampling, npixel_kernel)


def frac_coord(npixel, kernel_oversampling, p):
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


def frac_coords(shape, kernel_oversampling, xycoords):
    """Compute grid coordinates and fractional values for convolutional gridding

    :param shape: (height,width) grid shape
    :param kernel_oversampling: Oversampling factor
    :param xycoords: array of (x,y) coordinates in range [-.5,.5[
    """
    _, _, h, w = shape  # NB order (height,width) to match numpy!
    y, yf = frac_coord(h, kernel_oversampling, xycoords[..., 1])
    x, xf = frac_coord(w, kernel_oversampling, xycoords[..., 0])
    return x, xf, y, yf


def fixed_kernel_degrid(kernel, vshape, uvgrid, uv, frequency, polarisation, uvscale, vpolmap = lambda pol: pol,
                        vchanmap = lambda frequency: 0):
    """Convolutional degridding with frequency and polarisation independent

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param kernel: Oversampled convolution kernel
    :param vshape: Shape of visibility
    :param uvgrid:   The uv plane to de-grid from
    :param uv: fractional uv coordinates in range[-0.5,0.5[
    :param uvscale: scaling for each channel
    :returns: Array of visibilities.
    """
    kernel_oversampling, _, gh, gw = kernel.shape
    assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
    assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
    inchan, inpol, ny, nx = uvgrid.shape
    nvis, vnchan, vnpol = vshape
    assert vnpol == inpol, "Number of polarizations must be the same"
    vis = numpy.zeros([nvis], dtype='complex')
    wt  = numpy.zeros([nvis])
    
    ichan = vchanmap(frequency)
    ipol = vpolmap(polarisation)
    # Calculate grid point and fractional part for scaled u and v
    y, yf = frac_coord(uvgrid.shape[2], kernel_oversampling, uvscale[1] * uv[..., 1])
    x, xf = frac_coord(uvgrid.shape[3], kernel_oversampling, uvscale[0] * uv[..., 0])
    # Now calculate slices for the footprint of each sample
    slicey = []
    slicex = []
    for xx in x:
        slicex.append(slice((xx - gw // 2), (xx + (gw + 1) // 2)))
    for yy in y:
        slicey.append(slice((yy - gh // 2), (yy + (gh + 1) // 2)))
    coords = slicex, xf, slicey, yf
    # Now we can degrid the points and accumulate weight as well
    vis[...] = [
        numpy.sum(uvgrid[ichan, ipol, sly, slx] * kernel[yf, xf, :, :])
        for slx, xf, sly, yf in zip(*coords)
        ]
    wt[...] = [numpy.sum(kernel[yf, xf, :, :].real) for x, xf, y, yf in zip(*coords)]
    vis[numpy.where(wt > 0)] = vis[numpy.where(wt > 0)] / wt[numpy.where(wt > 0)]
    vis[numpy.where(wt < 0)] = 0.0
    return numpy.array(vis)

def gridder(uvgrid, vis, xs, ys, kernel=numpy.ones((1,1)), kernel_ixs=None):
    """Grids visibilities at given positions. Convolution kernels are selected per
    visibility using ``kernel_ixs``.

    :param uvgrid: Grid to update (two-dimensional :class:`complex` array)
    :param vis: CompressedVisibility values (one-dimensional :class:`complex` array)
    :param xs: CompressedVisibility position (one-dimensional :class:`int` array)
    :param ys: CompressedVisibility values (one-dimensional :class:`int` array)
    :param kernel: Convolution kernel (minimum two-dimensional :class:`complex` array).
      If the kernel has more than two dimensions, additional indices must be passed
      in ``kernel_ixs``. Default: Fixed one-pixel kernel with value 1.
    :param kernel_ixs: Map of visibilities to kernel indices (maximum two-dimensional :class:`int` array).
      Can be omitted if ``kernel`` requires no indices, and can be one-dimensional
      if only one index is needed to identify kernels
    """

    if kernel_ixs is None:
        kernel_ixs = numpy.zeroes((len(vis),0))
    else:
        kernel_ixs = numpy.array(kernel_ixs)
        if len(kernel_ixs.shape) == 1:
            kernel_ixs = kernel_ixs.reshape(len(kernel_ixs), 1)

    gh, gw = kernel.shape[-2:]
    for v, x, y, kern_ix in zip(vis, xs, ys, kernel_ixs):
        uvgrid[y:y+gh, x:x+gw] += kernel[tuple(kern_ix)] * v


def fixed_kernel_grid(kernel, uvgrid, uv, frequency, polarisation, uvscale, vis, visweights,
                      vpolmap = lambda pol: pol, vchanmap = lambda frequency: 0):
    """Grid after convolving with frequency and polarisation independent gcf

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param kernel: Oversampled convolution kernel
    :param uvgrid: Grid to add to
    :param uv: UVW positions
    :param uvscale: Scaling for each axis (u,v) for each channel
    :param vis: CompressedVisibility values
    :param vis: CompressedVisibility weights
    :returns: uv grid[nchan, npol, ny, nx], sumwt[nchan, npol]
    """
    
    kernel_oversampling, _, gh, gw = kernel.shape
    assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
    assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
    inchan, inpol, ny, nx = uvgrid.shape
    nvis, vnchan, vnpol = vis.shape
    assert vnpol == inpol, "Number of polarizations must be the same"
    wtgrid = numpy.zeros(uvgrid.shape, dtype='float')
    sumwt = numpy.zeros([inchan, inpol])
    ichan = vchanmap(frequency)
    ipol = vpolmap(polarisation)
    y, yf = frac_coord(ny, kernel_oversampling, uvscale[1] * uv[..., 1])
    x, xf = frac_coord(nx, kernel_oversampling, uvscale[0] * uv[..., 0])
    slicey = []
    slicex = []
    for xx in x:
        slicex.append(slice((xx - gw // 2), (xx + (gw + 1) // 2)))
    for yy in y:
        slicey.append(slice((yy - gh // 2), (yy + (gh + 1) // 2)))
    coords = slicex, xf, slicey, yf
    wts = visweights[...]
    viswt = vis[...] * visweights[...]
    coords = ichan, ipol, slicex, xf, slicey, yf
    for v, vwt, ic, ip, slx, xf, sly, yf in zip(viswt, wts, *coords):
        uvgrid[ic, ip, sly, slx] += kernel[yf, xf, :, :] * v
        wtgrid[ic, ip, sly, slx] += kernel[yf, xf, :, :].real * vwt
    sumwt[ichan, ipol] += numpy.sum(wtgrid[ichan, ipol, ...])
    
    return uvgrid, sumwt


def box_grid(kernel, uvgrid, uv, frequency, polarisation, uvscale, vis, visweights, vpolmap = lambda pol: pol,
             vchanmap = lambda chan: chan):
    """Grid with a box function

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param kernel: Oversampled convolution kernel
    :param uvgrid: Grid to add to
    :param uv: UVW positions
    :param uvscale: Scaling for each axis (u,v) for each channel
    :param vis: CompressedVisibility values
    :param vis: CompressedVisibility weights
    :param chanmap: function to map visibility channels to image channels
    """
    
    inchan, inpol, ny, nx = uvgrid.shape
    nvis, vnchan, vnpol = vis.shape
    assert vnpol == inpol, "Number of polarizations must be the same"
    wtgrid = numpy.zeros(uvgrid.shape, dtype='float')
    sumwt = numpy.zeros([inchan, inpol])
    ichan = vchanmap(frequency)
    ipol = vpolmap(polarisation)
    y, _ = frac_coord(ny, 1, uvscale[1] * uv[..., 1])
    x, _ = frac_coord(nx, 1, uvscale[0] * uv[..., 0])
    coords = frequency, polarisation, x, y
    wts = visweights[...]
    viswt = vis[...] * visweights[...]
    for v, vwt, ic, ip, x, y in zip(viswt, wts, *coords):
        uvgrid[ic, ip, y, x] += v
        wtgrid[ic, ip, y, x] += vwt
    sumwt = numpy.sum(wtgrid, axis=(2,3))
    
    return uvgrid, sumwt


def weight_gridding(shape, uv, uvscale, visweights, weighting='uniform',
                    vpolmap = lambda pol: pol, vchanmap = lambda chan: chan):
    """Reweight data using one of a number of algorithms

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param shape:
    :param uv: UVW positions
    :param uvscale: Scaling for each axis (u,v) for each channel
    :param vis: CompressedVisibility values
    :param visweights: CompressedVisibility weights
    :param weighting: Weighting algorithm (natural|uniform) (uniform)
    :param vpolmap = lambda pol: pol, vchanmap: function to map visibility channels to image channels
    """
    densitygrid = numpy.zeros(shape)
    density = numpy.zeros_like(visweights)
    if weighting == 'uniform':
        log.info("weight_gridding: Performing uniform weighting")
        inchan, inpol, ny, nx = shape
        nvis, vnchan, vnpol = visweights.shape
        # Add all visibility points to a float grid
        for vchan in range(vnchan):
            ichan = vpolmap = lambda pol: pol, vchanmap(vchan)
            y, _ = frac_coord(ny, 1, uvscale[1] * uv[..., 1])
            x, _ = frac_coord(nx, 1, uvscale[0] * uv[..., 0])
            coords = x, y
            for pol in range(vnpol):
                wts = visweights[..., vchan, pol]
                for wt, x, y, in zip(wts, *coords):
                    densitygrid[ichan, pol, y, x] += wt
                # for i in range(visweights[..., chan, pol].shape[0]):
                #     densitygrid[chan, pol, y[i], x[i]] += wts[i]

        # Normalise each visibility weight to sum to one in a grid cell
        newvisweights = numpy.zeros_like(visweights)
        for vchan in range(vnchan):
            ichan = vpolmap = lambda pol: pol, vchanmap(vchan)
            coords = frac_coord(shape[3], 1, uvscale[1] * uv[..., 1]), \
                     frac_coord(shape[2], 1, uvscale[0] * uv[..., 0])
            for pol in range(vnpol):
                wts = visweights[..., vchan, pol]
                for wt, x, y in zip(wts, *coords):
                    density[..., vchan, pol] += densitygrid[ichan, pol, y, x]
                    newvisweights[..., vchan, pol][densitygrid[ichan, pol, y, x] > 0.0] = \
                        wt / densitygrid[ichan, pol, y, x][densitygrid[ichan, pol, y, x] > 0.0]
        return newvisweights, density, densitygrid
    else:
        return visweights, density, densitygrid
