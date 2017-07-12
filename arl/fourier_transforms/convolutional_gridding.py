""" Imaging is based on used of the FFT to perform Fourier transforms efficiently. Since the observed visibility data
do not arrive naturally on grid points, the sampled points are resampled on the FFT grid using a convolution function to
smear out the sample points. The resulting grid points are then FFT'ed. The result can be corrected for the gridding
convolution function by division in the image plane of the transform.

This approach may be extended to include image plane effect such as the w term and the antenna/station primary beam.

This module contains functions for performing the gridding process and the inverse degridding process.
"""

from __future__ import division

import logging

import scipy.special

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


def coordinates2Offset(npixel: int, cx: int, cy: int):
    """Two dimensional grids of coordinates centred on an arbitrary point.
    
    This is used for A and w beams.

    1. a step size of 2/npixel and
    2. (0,0) at pixel (cx, cy,floor(n/2))
    """
    if cx is None:
        cx = npixel//2
    if cy is None:
        cy = npixel//2
    mg = numpy.mgrid[0:npixel,0:npixel]
    return ((mg[0]-cy)/npixel, (mg[1]-cx)/npixel)


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



def grdsf(nu):
    """Calculate PSWF using an old SDE routine re-written in Python

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


def w_beam(npixel, field_of_view, w, cx=None, cy=None, remove_shift=False):
    """ W beam, the fresnel diffraction pattern arising from non-coplanar baselines
    
    :param npixel: Size of the grid in pixels
    :param field_of_view: Field of view
    :param w: Baseline distance to the projection plane
    :param cx: location of delay centre def :npixel//2
    :param cy: location of delay centre def :npixel//2
    :param remove_shift: Remove overall phase shift
    :returns: npixel x npixel array with the far field
    """
    if cx is None:
        cx = npixel // 2
    if cy is None:
        cy = npixel // 2
    m, l = coordinates2Offset(npixel, cx, cy)
    m *= field_of_view
    l *= field_of_view
    r2 = l ** 2 + m ** 2
    n2 = 1.0 - r2
    ph = numpy.zeros_like(n2)
    ph[n2 < 1.0] = w * (1 - numpy.sqrt(1.0 - r2[n2 < 1.0]))
    cp = numpy.zeros_like(n2, dtype='complex')
    cp[n2 < 1.0] = numpy.exp(-2j * numpy.pi * ph[n2 < 1.0])
    cp[r2==0] = 1.0+0j
    # Correct for linear phase shift in faceting
    if remove_shift:
        cp /= cp[npixel//2, npixel//2]
    return cp


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


def w_kernel(field_of_view, w, npixel_farfield, npixel_kernel, kernel_oversampling, cx, cy, remove_shift):
    """ The middle s pixels of W convolution kernel. (W-KERNel-Aperture-Function)

    :param remove_shift:
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
    wbeamarray= w_beam(npixel_farfield, field_of_view, w, cx, cy, remove_shift=remove_shift) / gcf
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
    assert numpy.array(p >= -0.5).all() and numpy.array(p < 0.5).all(), "Cellsize is too large: uv overflows grid uv= %s" % str(p)
    x = npixel // 2 + p * npixel
    flx = numpy.floor(x + 0.5 / kernel_oversampling)
    fracx = numpy.around((x - flx) * kernel_oversampling)
    return flx.astype(int), fracx.astype(int)


def convolutional_degrid(kernel_list, vshape, uvgrid, vuvwmap, vfrequencymap, vpolarisationmap):
    """Convolutional degridding with frequency and polarisation independent

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param kernels: list of oversampled convolution kernel
    :param vshape: Shape of visibility
    :param uvgrid:   The uv plane to de-grid from
    :param vuvwmap: function to map uvw to grid fractions
    :param vfrequencymap: function to map frequency to image channels
    :param vpolarisationmap: function to map polarisation to image polarisation
    :returns: Array of visibilities.
    """
    kernel_indices, kernels = kernel_list
    kernel_oversampling, _, gh, gw = kernels[0].shape
    assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
    assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
    inchan, inpol, ny, nx = uvgrid.shape
    vnpol = vshape[-1]
    nvis = vshape[0]
    vis = numpy.zeros([nvis, vnpol], dtype='complex')
    wt  = numpy.zeros([nvis, vnpol])

    # uvw -> fraction of grid mapping
    y, yf = frac_coord(ny, kernel_oversampling, vuvwmap[:,1])
    x, xf = frac_coord(nx, kernel_oversampling, vuvwmap[:,0])

    # Now calculate slices for the footprint of each sample
    slicey = []
    slicex = []
    for xx in x:
        slicex.append(slice((xx - gw // 2), (xx + (gw + 1) // 2)))
    for yy in y:
        slicey.append(slice((yy - gh // 2), (yy + (gh + 1) // 2)))

    coords = kernel_indices, list(vfrequencymap), xf, yf
    for pol in range(vnpol):
        vis[...,pol] = [
            numpy.sum(uvgrid[ic, pol, sly, slx] *
                      numpy.conjugate(kernels[kernel_index][yf, xf, :, :]))
            for slx, sly, kernel_index, ic, xf, yf in zip(slicex, slicey, *coords)
            ]

        wt[...,pol] = [
            numpy.sum(kernels[kernel_index][yf, xf, :, :].real)
            for kernel_index, ic, xf, yf in zip(*coords)
            ]
    vis[numpy.where(wt > 0)] = vis[numpy.where(wt > 0)] / wt[numpy.where(wt > 0)]
    vis[numpy.where(wt < 0)] = 0.0
    return numpy.array(vis)

def gridder(uvgrid, vis, xs, ys, kernel=numpy.ones((1,1)), kernel_ixs=None):
    """Grids visibilities at given positions. Convolution kernels are selected per
    visibility using ``kernel_ixs``.

    :param uvgrid: Grid to update (two-dimensional :class:`complex` array)
    :param vis: Visibility values (one-dimensional :class:`complex` array)
    :param xs: Visibility position (one-dimensional :class:`int` array)
    :param ys: Visibility values (one-dimensional :class:`int` array)
    :param kernel: Convolution kernel (minimum two-dimensional :class:`complex` array).
      If the kernel has more than two dimensions, additional indices must be passed
      in ``kernel_ixs``. Default: Fixed one-pixel kernel with value 1.
    :param kernel_ixs: Map of visibilities to kernel indices (maximum two-dimensional :class:`int` array).
      Can be omitted if ``kernel`` requires no indices, and can be one-dimensional
      if only one index is needed to identify kernels
    """

    if kernel_ixs is None:
        kernel_ixs = numpy.zeros((len(vis),0))
    else:
        kernel_ixs = numpy.array(kernel_ixs)
        if len(kernel_ixs.shape) == 1:
            kernel_ixs = kernel_ixs.reshape(len(kernel_ixs), 1)

    gh, gw = kernel.shape[-2:]
    for v, x, y, kern_ix in zip(vis, xs, ys, kernel_ixs):
        uvgrid[y:y+gh, x:x+gw] += kernel[tuple(kern_ix)] * v


def convolutional_grid(kernel_list, uvgrid, vis, visweights, vuvwmap, vfrequencymap, vpolarisationmap):
    """Grid after convolving with frequency and polarisation independent gcf

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param kernels: List of versampled convolution kernels
    :param uvgrid: Grid to add to
    :param vis: Visibility values
    :param visweights: Visibility weights
    :param vuvwmap: map uvw to grid fractions
    :param vfrequencymap: map frequency to image channels
    :param vpolarisationmap: map polarisation to image polarisation
    :returns: uv grid[nchan, npol, ny, nx], sumwt[nchan, npol]
    """
    
    kernel_indices, kernels = kernel_list
    kernel_oversampling, _, gh, gw = kernels[0].shape
    assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
    assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
    inchan, inpol, ny, nx = uvgrid.shape

    # Construct output grids (in uv space)
    sumwt = numpy.zeros([inchan, inpol])
    
    # uvw -> fraction of grid mapping
    # vuvwmap is a list of generators, one per axis so we need to
    # select a generator and then instantiate it by making it into
    # a list, and finally use the values. This defers filling out
    # the arrays until they are needed.
    y, yf = frac_coord(ny, kernel_oversampling, vuvwmap[:,1])
    x, xf = frac_coord(nx, kernel_oversampling, vuvwmap[:,0])
    
    # Construct all the slices that we will need: trade off memory to
    # simplify loop
    slicey = []
    slicex = []
    for xx in x:
        slicex.append(slice((xx - gw // 2), (xx + (gw + 1) // 2)))
    for yy in y:
        slicey.append(slice((yy - gh // 2), (yy + (gh + 1) // 2)))
        
    # Now we can loop over all rows
    wts = visweights[...]
    viswt = vis[...] * visweights[...]
    
    npol = vis.shape[-1]
    
    # About 57k samples per second for standard kernel so about 2.5 million CMACs per second
    coords = kernel_indices, list(vfrequencymap), xf, yf
    for pol in range(npol):
        for v, vwt, slx, sly, kind, ic,  xf, yf in zip(viswt[...,pol], wts[...,pol], slicex,  slicey, *coords):
            uvgrid[ic, pol, sly, slx] += kernels[kind][yf, xf, :, :] * v
            sumwt[ic, pol] += numpy.sum(kernels[kind][yf, xf, :, :].real * vwt)

    return uvgrid, sumwt


def weight_gridding(shape, visweights, vuvwmap, vfrequencymap, vpolarisationmap, weighting='uniform'):
    """Reweight data using one of a number of algorithms

    :param shape:
    :param visweights: Visibility weights
    :param vuvwmap: map uvw to grid fractions
    :param vfrequencymap: map frequency to image channels
    :param vpolarisationmap: map polarisation to image polarisation
    :param weighting: '' | 'uniform'
    :returns: visweights, density, densitygrid
    """
    densitygrid = numpy.zeros(shape)
    density = numpy.zeros_like(visweights)
    if weighting == 'uniform':
        log.info("weight_gridding: Performing uniform weighting")
        inchan, inpol, ny, nx = shape
        
        # uvw -> fraction of grid mapping
        y, yf = frac_coord(ny, 1.0, vuvwmap[:, 1])
        x, xf = frac_coord(nx, 1.0, vuvwmap[:, 0])
        wts = visweights[...]
        coords = list(vfrequencymap), x, y
        for pol in range(inpol):
            for vwt, ic, x, y in zip(wts, *coords):
                densitygrid[ic, pol, y, x] += vwt[..., pol]
    
        # Normalise each visibility weight to sum to one in a grid cell
        newvisweights = numpy.zeros_like(visweights)
        for pol in range(inpol):
            density[..., pol] += [densitygrid[ic, pol, x, y] for ic, x, y in zip(*coords)]
        newvisweights[density>0.0] = visweights[density>0.0]/density[density>0.0]
        return newvisweights, density, densitygrid
    else:
        return visweights, None, None
