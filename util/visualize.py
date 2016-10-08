
from crocodile.synthesis import coordinateBounds

import numpy
import matplotlib.pyplot as pl
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d

def show_image(img, name, field_of_view, norm=None, extra_dep=None):
    """Visualise quadratic image in the (L,M) plane (directional
    cosines). We assume (0,0) to be at the image center.

    :param img: Data to visualise as a two-dimensional numpy array
    :param name: Function name to show in the visualisation header
    :param field_of_view: Size of the image in radians. We will assume the
       image to spans coordinates [field_of_view/2;field_of_view/2[ in both L and M.
    :param extra_dep: Extra functiona parameters to add to the
       title. Purely cosmetic.
    """

    # Determine size of image.
    size = img.shape[0]
    lm_lower, lm_upper = coordinateBounds(size)
    lm_lower = (lm_lower-1./size/2)*field_of_view
    lm_upper = (lm_upper+1./size/2)*field_of_view
    extent = (lm_lower, lm_upper, lm_lower, lm_upper)

    # Format title
    title = "%s(l,m%s)" % (name, ','+extra_dep if extra_dep is not None else "")

    # Determine normalisation for image.
    if norm is not None:
        norm = colors.Normalize(vmin=-norm, vmax=norm, clip=True)
    else:
        norm = None

    if numpy.any(numpy.iscomplex(img)):
        pl.subplot(121)
    else:
        pl.subplot(111)
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
    uv_lower, uv_upper = coordinateBounds(size)
    uv_lower = (uv_lower-1./size/2)*lam
    uv_upper = (uv_upper+1./size/2)*lam
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


# from http://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
# by CT Zhu
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def make_arrow(ax, source, target, color, name=None, scale=20, lw=3):
    xs, ys, zs = numpy.transpose((source, target))
    ax.add_artist(Arrow3D(xs, ys, zs, mutation_scale=scale, lw=lw, arrowstyle="-|>", color=color))
    if name is not None:
        ax.text(target[0]+0.03, target[1], target[2], name, color=color)


def circular_to_xyz(lon, lat):
    """Circular coordinate transformation appropriate for visualisation"""
    return numpy.array((numpy.sin(lon) * numpy.cos(lat),
                        numpy.cos(lon) * numpy.cos(lat),
                        numpy.sin(lat)))

def visualise_uvw(latitude, hour_angle, declination):
    """Shows a visualisation for the UVW coordinate system for an earth
    observer's UVW coordinate system pointing towards a certain local
    celestial coordinate.

    :param latitude: Latitude of the observer. Should be 0-90 degrees.
    :param hour_angle: Hour angle of the source. Should be -90-90 degrees.
    :param declination: Declination of the source. Should be -90-90 degrees.
    """

    def draw():
        make_arrow(ax, [0,0,0],[0,0,1.1], "black", "Earth axis (towards celestial north)")
        lons = numpy.linspace(-numpy.pi/4, numpy.pi/4, 10)
        lats = numpy.linspace(0, numpy.pi/2, 10)
        x, y, z = circular_to_xyz(numpy.outer(lons, numpy.ones(len(lats))),
                                  numpy.outer(numpy.ones(len(lons)), lats))
        ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, alpha=0.4)
        obs_x, obs_y, obs_z = obs = circular_to_xyz(0, numpy.radians(latitude))
        ax.plot([0, obs_x], [0, obs_y], [0, obs_z], color="black", lw=3)
        ax.text(obs_x+0.03, obs_y, obs_z, "Observer", color="black")
        wdir = circular_to_xyz(numpy.radians(hour_angle), numpy.radians(declination))
        vdir = circular_to_xyz(numpy.radians(hour_angle), numpy.radians(declination+90))
        udir = numpy.cross(vdir, wdir)
        make_arrow(ax, obs, obs+wdir/3, "red", "w (towards phase centre)")
        make_arrow(ax, obs, obs+vdir/3, "red", "v")
        make_arrow(ax, obs, obs+udir/3, "red", "u")
        make_arrow(ax, obs, obs+[0,0,0.2], "black", "")

    fig = pl.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.view_init(elev=20., azim=-20.)
    ax.set_title("earth view")
    draw()
    ax = fig.add_subplot(122, projection='3d')
    ax.view_init(elev=declination, azim=90.-hour_angle)
    ax.set_title("view from phase centre")
    draw()

def visualise_lmn(hour_angle, declination):
    # Swap X and Y to get a right-handed coordinate system
    def trans(coo):
        x,y,z = coo
        return y, x, z
    def draw(ax):
        ax.set_xlabel('y [$1$]'); ax.set_ylabel('x [$1$]'); ax.set_zlabel('z [$1$]')
        make_arrow(ax, trans([0,0,0]),trans([0,0,1.1]), "black", "Celestial north")
        make_arrow(ax, trans([0,0,0]),trans([1.1,0,0]), "black", "Geographical east")
        lons = numpy.linspace(0, numpy.pi/2, 20)
        lats = numpy.linspace(0, numpy.pi/2, 20)
        x, y, z = circular_to_xyz(numpy.outer(lons, numpy.ones(len(lats))),
                                  numpy.outer(numpy.ones(len(lons)), lats))
        ax.plot_surface(y, x, z, rstride=1, cstride=1, linewidth=0, alpha=0.4, color='white')
        t_x, t_y, t_z = ndir = circular_to_xyz(numpy.radians(-hour_angle), numpy.radians(declination))
        make_arrow(ax, trans((0,0,0)),     trans((t_x,0,0)),     color="gray", lw=3)
        make_arrow(ax, trans((t_x,0,0)),   trans((t_x,t_y,0)),   color="gray", lw=3)
        make_arrow(ax, trans((t_x,t_y,0)), trans((t_x,t_y,t_z)), color="gray", lw=3)
        make_arrow(ax, trans((0,0,0)),     trans((t_x,t_y,t_z)), color="black", name="phase centre", lw=3)
        mdir = circular_to_xyz(numpy.radians(-hour_angle), numpy.radians(declination+90))
        ldir = numpy.cross(ndir, mdir)
        make_arrow(ax, trans(ndir), trans(ndir+ndir/3), "red", "n")
        make_arrow(ax, trans(ndir), trans(ndir+ldir/3), "red", "l")
        make_arrow(ax, trans(ndir), trans(ndir+mdir/3), "red", "m")
        ax.set_title("Phase centre at $(%.2f,%.2f,%.2f)$" % (t_x, t_y, t_z))
    fig = pl.figure()
    ax = fig.add_subplot(121, projection='3d')
    draw(ax)
    ax.view_init(elev=35, azim=25)
    pl.show()
