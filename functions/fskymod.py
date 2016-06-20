# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

from collections import namedtuple

import numpy

from astropy.coordinates import SkyCoord

from functions.fimage import fimage, fimage_from_fits
from functions.fcomp import fcomp, fcomp_construct


def fskymod():
    fsm = namedtuple("fskymod", ['images', 'components'])
    fsm.images = []
    fsm.components = []
    return fsm


def fskymod_filter(fsm: fskymod, **kwargs):
    print("fskymod: No filter implemented yet")
    return fsm


def fskymod_add(fsm1: fskymod, fsm2: fskymod):
    """
    Add two configurations together
    :param fsm1:
    :param fsm2:
    :return:
    """
    fsm=fskymod()
    fsm.images = [fsm1.images, fsm2.images]
    fsm.components = [fsm1.components, fsm2.components]

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
    m31im = fskymod()
    m31im.images.append(m31image)
    flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
    comp = fcomp_construct(direction, flux, shape='Point', name="Mysource")
    m31comp = fskymod()
    m31comp.components.append(comp)
    m31added=fskymod_add(m31im, m31comp)
    print(dir(m31added))
