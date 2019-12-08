""" Imaging is based on used of the FFT to perform Fourier transforms efficiently. Since the observed visibility data_models
do not arrive naturally on grid points, the sampled points are resampled on the FFT grid using a convolution function to
smear out the sample points. The resulting grid points are then FFT'ed. The result can be corrected for the griddata
convolution function by division in the image plane of the transform.

This approach may be extended to include image plane effect such as the w term and the antenna/station primary beam.

This module contains functions for performing the griddata process and the inverse degridding process.

All grids and images are considered quadratic and centered around
`npixel//2`, where `npixel` is the pixel width/height. This means that `npixel//2` is
the zero frequency for FFT purposes, as is convention. Note that this
means that for even `npixel` the grid is not symmetrical, which means that
e.g. for convolution kernels odd image sizes are preferred.

This is implemented for reference in `coordinates`/`coordinates2`. Some noteworthy properties:
- `ceil(field_of_view * lam)` gives the image size `npixel` in pixels
- `lam * coordinates2(npixel)` yields the `u,v` grid coordinate system
- `field_of_view * coordinates2(npixel)` yields the `l,m` image coordinate system (radians, roughly)

"""

__all__ = ['w_beam', 'weight_gridding', 'visibility_recentre']

import logging

import numpy

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


def coordinates(npixel: int):
    """ 1D array which spans [-.5,.5[ with 0 at position npixel/2
    
    """
    return (numpy.arange(npixel) - npixel // 2) / npixel


def coordinates2(npixel: int):
    """Two dimensional grids of coordinates spanning -1 to 1 in each dimension

    1. a step size of 2/npixel and
    2. (0,0) at pixel (floor(n/2),floor(n/2))
    """
    return (numpy.mgrid[0:npixel, 0:npixel] - npixel // 2) / npixel


def coordinates2Offset(npixel: int, cx: int, cy: int, quadrant=False):
    """Two dimensional grids of coordinates centred on an arbitrary point.

    This is used for A and w beams.

    1. a step size of 2/npixel and
    2. (0,0) at pixel (cx, cy,floor(n/2))
    """
    if cx is None:
        cx = npixel // 2
    if cy is None:
        cy = npixel // 2
    if quadrant == False:
        mg = numpy.mgrid[0:npixel, 0:npixel]
    else:
        # If npixel is even, we should create a grid with npixel//2+1
        mg = numpy.mgrid[0:npixel//2+1, 0:npixel//2+1]
    return (mg[0] - cy) / npixel, (mg[1] - cx) / npixel

def grdsf(nu):
    """Calculate PSWF using an old SDE routine re-written in Python

    Find Spheroidal function with M = 6, alpha = 1 using the rational
    approximations discussed by Fred Schwab in 'Indirect Imaging'.
    This routine was checked against Fred's SPHFN routine, and agreed
    to about the 7th significant digit.
    The griddata function is (1-NU**2)*GRDSF(NU) where NU is the distance
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
    part[(nu >= 0.75) & (nu <= 1.0)] = 1
    nuend[(nu >= 0.0) & (nu < 0.75)] = 0.75
    nuend[(nu >= 0.75) & (nu <= 1.0)] = 1.0

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

    # Return the griddata function and the grid correction function
    return grdsf, (1 - nu ** 2) * grdsf


def w_beam(npixel, field_of_view, w, cx=None, cy=None, remove_shift=False):
    """ W beam, the fresnel diffraction pattern arising from non-coplanar baselines
    
    :param npixel: Size of the grid in pixels
    :param field_w_beamof_view: Field of view
    :param w: Baseline distance to the projection plane
    :param cx: location of delay centre def :npixel//2
    :param cy: location of delay centre def :npixel//2
    :param remove_shift: Remove overall phase shift at the centre of the image
    :return: npixel x npixel array with the far field
    """
    if cx is None:
        cx = npixel // 2
    if cy is None:
        cy = npixel // 2

    # Original codes
    # ly, mx = coordinates2Offset(npixel, cx, cy)
    # r2 = field_of_view**2*(ly ** 2 + mx ** 2)
    # ph = numpy.zeros_like(r2)
    # ph[r2 < 1.0] = -2 * numpy.pi * w * (1 - numpy.sqrt(1.0 - r2[r2 < 1.0]))
    # cp = numpy.zeros_like(r2, dtype='complex')
    # cp[r2 < 1.0] = numpy.exp(1j * ph[r2 < 1.0])
    # cp[r2 == 0] = 1.0 + 0j
    # if remove_shift:
    #     cp /= cp[npixel // 2, npixel // 2]

    # numpy.putmask
    # ly, mx = coordinates2Offset(npixel, cx, cy)
    # r2 = field_of_view**2*(ly ** 2 + mx ** 2)
    # ph = numpy.zeros_like(r2)
    # m = r2 < 1.0
    # cp = numpy.zeros_like(r2, dtype='complex')
    # numpy.putmask(ph, m, -2 * numpy.pi * w * (1 - numpy.sqrt(1.0 - r2)))
    # numpy.putmask(cp, m, numpy.exp(1j * ph))
    # numpy.putmask(cp, r2 == 0, 1.0 + 0j)
    # if remove_shift:
    #     cp /= cp[npixel // 2, npixel // 2]

    # numpy.putmask - 2
    # ly, mx = coordinates2Offset(npixel, cx, cy)
    # r2 = field_of_view ** 2 * (ly ** 2 + mx ** 2)
    # ph = -2 * numpy.pi * w * (1 - numpy.sqrt(1.0 - r2))
    # numpy.putmask(ph, r2 >= 1.0, 0)
    # cp = numpy.zeros_like(r2, dtype='complex')
    # cp = numpy.exp(1j * ph)
    # numpy.putmask(cp, r2 >= 1.0, 0 + 0j)
    # numpy.putmask(cp, r2 == 0, 1.0 + 0j)
    # if remove_shift:
    #     cp /= cp[npixel // 2, npixel // 2]

    # SubArray Copy Symmetrically
    ly, mx = coordinates2Offset(npixel, cx, cy, quadrant=True)
    r2 = field_of_view ** 2 * (ly ** 2 + mx ** 2)
    ph = -2 * numpy.pi * w * (1 - numpy.sqrt(1.0 - r2))
    numpy.putmask(ph, r2 >= 1.0, 0)
    cp = numpy.zeros_like(r2, dtype='complex')
    cp = numpy.exp(1j * ph)
    numpy.putmask(cp, r2 >= 1.0, 0 + 0j)
    numpy.putmask(cp, r2 == 0, 1.0 + 0j)
    # Correct for linear phase shift in faceting
    if remove_shift:
        cp /= cp[-1, -1]

    cp = numpy.pad(cp, ((0, int(cx) + npixel % 2 - 1), (0, int(cy) + npixel % 2 - 1)), 'reflect')

    # assert((cp==cp1).all())

    return cp


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
    assert numpy.array(p >= -0.5).all() and numpy.array(
        p < 0.5).all(), "Cellsize is too large: uv overflows grid uv= %s" % str(p)
    x = npixel // 2 + p * npixel
    flx = numpy.floor(x + 0.5 / kernel_oversampling)
    fracx = numpy.around((x - flx) * kernel_oversampling)
    return flx.astype(int), fracx.astype(int)


def weight_gridding(shape, visweights, vuvwmap, vfrequencymap, vpolarisationmap=None, weighting='uniform'):
    """Reweight data using one of a number of algorithms

    :param shape:
    :param visweights: Visibility weights
    :param vuvwmap: map uvw to grid fractions
    :param vfrequencymap: map frequency to image channels
    :param vpolarisationmap: map polarisation to image polarisation
    :param weighting: '' | 'uniform'
    :return: visweights, density, densitygrid
    """
    densitygrid = numpy.zeros(shape, dtype='float')
    if weighting == 'uniform':
        log.debug("weight_gridding: Performing uniform weighting")
        inchan, inpol, ny, nx = shape

        wts = visweights[...]
        # uvw -> fraction of grid mapping
        for flip in [-1.0, 1.0]:
            y, yf = frac_coord(ny, 1.0, flip * vuvwmap[:, 1])
            x, xf = frac_coord(nx, 1.0, flip * vuvwmap[:, 0])
            coords = list(vfrequencymap), x, y
            for pol in range(inpol):
                for vwt, chan, x, y in zip(wts, *coords):
                    densitygrid[chan, pol, y, x] += vwt[..., pol]

        # Find the total weight per sample counting redundancies with other samples
        newvisweights = numpy.zeros_like(visweights)
        density = numpy.zeros_like(visweights)
        y, _ = frac_coord(ny, 1.0, vuvwmap[:, 1])
        x, _ = frac_coord(nx, 1.0, vuvwmap[:, 0])
        coords = list(vfrequencymap), x, y
        for pol in range(inpol):
            density[..., pol] += [densitygrid[chan, pol, y, x] for chan, x, y in zip(*coords)]

        # Normalise each visibility weight to sum to one in a grid cell
        if numpy.sum(density[:, 0] > 0.0) < visweights.shape[0]:
            log.warning("weight_gridding: Losing samples in weighting")

        newvisweights[density > 0.0] = visweights[density > 0.0] / density[density > 0.0]
        return newvisweights, density, densitygrid
    else:
        return visweights, None, None


def visibility_recentre(uvw, dl, dm):
    """ Compensate for kernel re-centering - see `w_kernel_function`.

    :param uvw: Visibility coordinates
    :param dl: Horizontal shift to compensate for
    :param dm: Vertical shift to compensate for
    :returns: Visibility coordinates re-centrered on the peak of their w-kernel
    """

    u, v, w = numpy.hsplit(uvw, 3)  # pylint: disable=unbalanced-tuple-unpacking
    return numpy.hstack([u - w * dl, v - w * dm, w])
