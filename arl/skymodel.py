# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

import numpy
from astropy.coordinates import SkyCoord

from arl.image import Image, image_from_fits
from arl.skycomponent import SkyComponent, create_skycomponent

"""
Functions that define and manipulate a model of the sky: images and sky components
"""


class SkyModel:
    """ A skymodel consisting of a list of images and a list of components
    """
    # TODO: Fill out SkyModel

    def __init__(self):
        self.images = []
        self.components = []


def skymodel_filter(sm: SkyModel, **kwargs):
    """Filter the sky model

    :param sm:
    :type SkyModel:
    :param kwargs:
    :returns: SkyModel
    """
    print("SkyModel: No filter implemented yet")
    return sm


def skymodel_add(sm1: SkyModel, sm2: SkyModel):
    """ Add two sky models together
    
    :param sm1:
    :type SkyModel:
    :param sm2:
    :type SkyModel:
    :returns: SkyModel
    """
    fsm = SkyModel()
    fsm.images = [sm1.images, sm2.images]
    fsm.components = [sm1.components, sm2.components]
    return fsm


def skymodel_from_image(im: Image):
    """ Create a skymodel from an image or image
    
    :param im:
    :type Image:
    :returns: SkyModel
    """
    sm = SkyModel()
    sm.images.append(im)
    return sm


def skymodel_add_image(sm: SkyModel, im: Image):
    """Add images to a sky model
    
    :param sm:
    :type SkyModel:
    :param im:
    :type Image:
    :returns: SkyModel
    """
    sm.images.append(im)
    return sm


def skymodel_from_component(comp: SkyComponent):
    """Create sky model from component
    
    :param comp:
    :type SkyComponent:
    :returns: SkyModel
    """
    sm = SkyModel()
    sm.components.append(comp)
    return sm


def skymodel_add_component(sm: SkyModel, comp: SkyComponent):
    """Add Component to a sky model
    
    :param sm:
    :type SkyModel:
    :param comp:
    :type SkyComponent:
    :returns: SkyModel
   """
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
    comp = create_skycomponent(flux, flux='Point', frequency=numpy.arange(5e6, 300e6, 1e7), shape='Point',
                               name="Mysource")
    m31comp = SkyModel()
    m31comp.components.append(comp)
    m31added = skymodel_add(m31im, m31comp)
