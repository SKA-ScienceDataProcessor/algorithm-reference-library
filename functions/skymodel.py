# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

from collections import namedtuple

import numpy

from astropy.coordinates import SkyCoord

from functions.image import image, image_from_fits
from functions.component import component


class skymodel():

    def __init__(self):
        self.images = []
        self.components = []


def skymodel_filter(fsm: skymodel, **kwargs):
    print("skymodel: No filter implemented yet")
    return fsm


def skymodel_add(fsm1: skymodel, fsm2: skymodel):
    """
    Add two configurations together
    :param fsm1:
    :param fsm2:
    :return:
    """
    fsm=skymodel()
    fsm.images = [fsm1.images, fsm2.images]
    fsm.components = [fsm1.components, fsm2.components]

def skymodel_from_image(images: image):
    """Add images
    """
    sm = skymodel()
    sm.images.append(images)


def skymodel_add_image(sm: skymodel, images: image):
    """Add images
    """
    sm.images.append(images)


def skymodel_from_component(comp: component):
    """Add Component
    """
    sm = skymodel()
    sm.components.append(comp)


def skymodel_add_component(sm: skymodel, comp: component):
    """Add Component
    """
    sm = skymodel()
    sm.components.append(comp)


if __name__ == '__main__':
    kwargs = {}
    m31image = skymodel_filter(image_from_fits("../data/models/m31.model.fits"), **kwargs)
    m31im = skymodel()
    m31im.images.append(m31image)
    flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
    comp = component_construct(direction, flux, shape='Point', name="Mysource")
    m31comp = skymodel()
    m31comp.components.append(comp)
    m31added=skymodel_add(m31im, m31comp)
    print(dir(m31added))
