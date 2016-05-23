
from crocodile.synthesis import ucsBounds

import numpy
import matplotlib.pyplot as pl
from matplotlib import colors

def show_image(img, name, theta, norm=None, extra_dep=None):
    """Visualise quadratic image in the (L,M) plane (directional
    cosines). We assume (0,0) to be at the image center.

    :param img: Data to visualise as a two-dimensional numpy array
    :param name: Function name to show in the visualisation header
    :param theta: Size of the image in radians. We will assume the
       image to spans coordinates [theta/2;theta/2[ in both L and M.
    :param extra_dep: Extra functiona parameters to add to the
       title. Purely cosmetic.
    """

    # Determine size of image.
    size = img.shape[0]
    lm_lower, lm_upper = ucsBounds(size)
    lm_lower = (lm_lower-1./size/2)*theta;
    lm_upper = (lm_upper+1./size/2)*theta;
    extent = (lm_lower, lm_upper, lm_lower, lm_upper)

    # Format title
    title = "%s(l,m%s)" % (name, ','+extra_dep if extra_dep is not None else "")

    # Determine normalisation for image.
    if norm is not None:
        norm = colors.Normalize(vmin=-norm, vmax=norm, clip=True)
    else:
        norm = None

    pl.subplot(121)
    pl.imshow(img.real, extent=extent, norm=norm, origin='lower')
    pl.title(r"$Re(%s)$" % title)
    pl.xlabel(r"L [$1$]"); pl.ylabel(r"M [$1$]")
    if norm is None: pl.colorbar(shrink=.4,pad=0.025)
    if numpy.any(numpy.iscomplex(img)):
        pl.subplot(122)
        pl.imshow(img.imag, extent=extent, norm=norm, origin='lower')
        pl.title(r"$Im(%s)$" % title)
        pl.xlabel(r"L [$1$]"); pl.ylabel(r"M [$1$]")
        if norm is None: pl.colorbar(shrink=.4,pad=0.025)
    pl.show()

def show_grid(grid, name, lam, norm=None, size=None):

    # Determine size of image. See above.
    size = grid.shape[0]
    uv_lower, uv_upper = ucsBounds(size)
    uv_lower = (uv_lower-1./size/2)*lam;
    uv_upper = (uv_upper+1./size/2)*lam;
    extent = (uv_lower, uv_upper, uv_lower, uv_upper)

    # Determine normalisation for image.
    if norm is not None:
        norm = colors.Normalize(vmin=-norm, vmax=norm, clip=True)
    else:
        norm = None

    # Draw.
    for plot, comp, data in [(121, "Re", grid.real), (122, "Im", grid.imag)]:
        pl.subplot(plot)
        pl.imshow(data, extent=extent, norm=norm, interpolation='nearest', origin='lower')
        pl.title("$%s(%s(u,v,w))$" % (comp, name))
        pl.xlabel(r"U [$\lambda$]")
        pl.ylabel(r"V [$\lambda$]")
        # Only show color bar if we don't use the standard normalisation.
        if norm is None: pl.colorbar(shrink=.4,pad=0.025)
    pl.show()
