# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

from collections import namedtuple

from functions.fimage import fimage, fimage_from_fits
from functions.fcomp import fcomp


def fskymod():
    fsm = namedtuple("fskymod", ['images', 'components'])
    fsm.images = []
    fsm.components = []
    return fsm


def fskymod_filter(fsm: fskymod, **kwargs):
    print("fskymod: No filter implemented yet")
    return fsm


def fskymod_from_fimage(images: fimage):
    """Add images
    """
    sm = fskymod()
    sm.images.append(images)


def fskymod_add_fimage(sm: fskymod, images: fimage):
    """Add images
    """
    sm.images.append(images)


def fskymod_from_fcomp(comp: fcomp):
    """Add Component
    """
    sm = fskymod()
    sm.components.append(comp)


def fskymod_add_fcomp(sm: fskymod, comp: fcomp):
    """Add Component
    """
    sm = fskymod()
    sm.components.append(comp)


if __name__ == '__main__':
    kwargs = {}
    m31image = fskymod_filter(fimage_from_fits("../data/models/m31.model.fits"), **kwargs)
    m31sm = fskymod()
    m31sm.images.append(m31image)
    print(dir(m31sm))
