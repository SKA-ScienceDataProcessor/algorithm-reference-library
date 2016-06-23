# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

import numpy
from astropy.coordinates import SkyCoord

from functions.image import Image, image_from_fits
from functions.skycomponent import SkyComponent


class SkyModel():
    """

    """

    def __init__(self):
        self.images = []
        self.components = []


def skymodel_filter(fsm: SkyModel, **kwargs):
    """

    :param fsm:
    :param kwargs:
    :return:
    """
    print("SkyModel: No filter implemented yet")
    return fsm


def skymodel_add(fsm1: SkyModel, fsm2: SkyModel):
    """
    Add two configurations together
    :param fsm1:
    :param fsm2:
    :return:
    """
    fsm = SkyModel()
    fsm.images = [fsm1.images, fsm2.images]
    fsm.components = [fsm1.components, fsm2.components]
    return fsm


def skymodel_from_image(images: Image):
    """Add images
    """
    sm = SkyModel()
    sm.images.append(images)
    return sm


def skymodel_add_image(sm: SkyModel, image: Image):
    """Add images
    """
    sm.images.append(image)
    return sm


def skymodel_from_component(comp: SkyComponent):
    """Add Component
    """
    sm = SkyModel()
    sm.components.append(comp)
    return sm


def skymodel_add_component(sm: SkyModel, comp: SkyComponent):
    """Add Component
    """
    sm = SkyModel()
    sm.components.append(comp)
    return sm


if __name__ == '__main__':
    import os

    os.chdir('../')
    print(os.getcwd())

    kwargs = {}
    m31image = skymodel_filter(image_from_fits("./data/models/M31.MOD"), **kwargs)
    m31im = SkyModel()
    m31im.images.append(m31image)
    flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
    comp = SkyComponent(direction, flux, shape='Point', name="Mysource")
    m31comp = SkyModel()
    m31comp.components.append(comp)
    m31added = skymodel_add(m31im, m31comp)
