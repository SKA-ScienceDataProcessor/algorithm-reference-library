# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
#
# Synthesise and Image interferometer data
"""Convolutional gridding support functions

All functions that involve convolutional gridding are kept here.
"""

from __future__ import division

from astropy.constants import c

from arl.fourier_transforms.convolutional_gridding import *

log = logging.getLogger("fourier_transforms.variable_kernels")


def standard_kernel_lambda(vis, shape, oversampling=8, support=3):
    """Return a lambda function to calculate the standard visibility kernel

    This is only required for testing speed versus fixed_kernel_grid.

    :param vis: visibility
    :param shape: tuple with 2D shape of grid
    :param oversampling: Oversampling factor
    :param support: Support of kernel
    :returns: Function to look up gridding kernel
    """
    sk = anti_aliasing_calculate(shape, oversampling, support)[1]
    return lambda row, chan=0: sk


def w_kernel_lambda(vis, shape, fov, oversampling=8, wstep=100.0, npixel_kernel=16, cache_size=10000):
    """Return a lambda function to calculate the w term visibility kernel

    This function is called once. It uses an LRU cache to hold the convolution kernels. As a result,
    initially progress is slow as the cache is filled. Then it speeds up.

    :param vis: visibility
    :param shape: tuple with 2D shape of grid
    :param fov: Field of view in radians
    :param oversampling: Oversampling factor
    :param support: Support of kernel
    :param wstep: Step in w between cached functions
    :param cache_size: Size of cache in items
    :returns: Function to look up gridding kernel as function of row, and cache
    """
    wmax = numpy.max(numpy.abs(vis.w)) * numpy.max(vis.frequency) / c.value
    log.debug("variable_kernels.w_kernel_lambda: Maximum w = %f wavelengths" % (wmax))
    
    def cached_on_w(w_integral):
        npixel_kernel_scaled = max(16, int(round(npixel_kernel*abs(w_integral*wstep)/wmax)))
        return w_kernel(field_of_view=fov, w=wstep * w_integral, npixel_farfield=shape[0],
                        npixel_kernel=npixel_kernel_scaled, kernel_oversampling=oversampling)
    
    lrucache = pylru.FunctionCacheManager(cached_on_w, size=cache_size)
    
    # The lambda function has arguments row and chan so any gridding function can only depend on those
    # parameters. Eventually we could extend that to include polarisation.
    return lambda row, chan=0: lrucache(int(round(vis.w[row] * vis.frequency[chan] / (c.value * wstep)))), lrucache


def variable_kernel_degrid(kernel_function, uvgrid, uv, uvscale):
    """Convolutional degridding with frequency and polarisation independent

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param kernel_function: Function to return oversampled convolution kernel for given row
    :param uvgrid:   The uv plane to de-grid from
    :param uv: fractional uv coordinates in range[-0.5,0.5[
    :param uvscale: scaling for each channel
    :param viscoords: list of visibility coordinates to use for kernel_function
    :returns: Array of visibilities.
    """
    nchan, npol, ny, nx = uvgrid.shape
    nvis, _ = uv.shape
    vis = numpy.zeros([nvis, nchan, npol], dtype='complex')
    wt = numpy.zeros([nvis, nchan, npol])
    for row in range(nvis):
        for chan in range(nchan):
            kernel = kernel_function(row, chan)
            kernel_oversampling, _, gh, gw = kernel.shape
            x, xf, y, yf = frac_coords(uvgrid.shape, kernel_oversampling, uv[row, :] * uvscale[chan])
            for pol in range(npol):
                vis[..., chan, pol] = numpy.sum(uvgrid[chan, pol, y - gh // 2: y + (gh + 1) // 2,
                                                x - gw // 2: x + (gw + 1) // 2] * kernel[yf, xf, :, :])
                wt[..., chan, pol] = numpy.sum(kernel[yf, xf, :, :].real)
        vis[numpy.where(wt > 0)] = vis[numpy.where(wt > 0)] / wt[numpy.where(wt > 0)]
    vis[numpy.where(wt < 0)] = 0.0
    return numpy.array(vis)


def variable_kernel_grid(kernel_function, uvgrid, uv, uvscale, vis, visweights):
    """Grid after convolving with frequency and polarisation independent gcf

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param kernel_function: Function to return oversampled convolution kernel for given row
    :param uvgrid: Grid to add to
    :param uv: UVW positions
    :param vis: Visibility values
    :param vis: Visibility weights
    """
    
    nchan, npol, ny, nx = uvgrid.shape
    nvis, _ = uv.shape
    for row in range(nvis):
        for chan in range(nchan):
            kernel = kernel_function(row, chan)
            kernel_oversampling, _, gh, gw = kernel.shape
            x, xf, y, yf = frac_coords(uvgrid.shape, kernel_oversampling, uv[row, :] * uvscale[chan])
            for pol in range(npol):
                viswt = vis[row, chan, pol] * visweights[row, chan, pol]
                uvgrid[chan, pol, (y - gh // 2):(y + (gh + 1) // 2), (x - gw // 2):(x + (gw + 1) // 2)] += \
                    kernel[yf, xf, :, :] * viswt
    
    return uvgrid
