# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

import numpy
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord

from arl.define_image import Image, create_image_from_fits


class SkyComponent:
    """ A single SkyComponent with direction, flux, shape, and parameters for the shape
    """
    # TODO: fill out SkyComponent


def create_skycomponent(direction: SkyCoord, flux: numpy.array, frequency: numpy.array, shape: str = 'Point',
                        param: dict = None, name: str = ''):
    """ A single SkyComponent with direction, flux, shape, and parameters for the shape

    :param direction:
    :type SkyCoord:
    :param flux:
    :type numpy.array:
    :param frequency:
    :type numpy.array:
    :param shape: 'Point' or 'Gaussian'
    :type str:
    :param name:
    :type str:
    :returns: SkyComponent
    """
    sc = SkyComponent()
    sc.direction = direction
    sc.frequency = frequency
    sc.name = name
    sc.flux = numpy.array(flux)
    sc.shape = shape
    sc.params = param
    sc.name = name
    return sc


def find_point_source(im: Image, **kwargs):
    """ Find components in Image, return SkyComponent, just find the peak for now

    :param im: Image to be searched
    :type Image:
    :returns: SkyComponent
    """
    # TODO: Implement full image fitting of components
    print("imaging.point_source_find: Finding components in Image")
    
    # Beware: The index sequencing is opposite in wcs and Python!
    locpeak = numpy.array(numpy.unravel_index((numpy.abs(im.data)).argmax(), im.data.shape))
    print("imaging.point_source_find: Found peak at pixel coordinates %s" % str(locpeak))
    w = im.wcs.sub(['longitude', 'latitude'])
    sc = pixel_to_skycoord(locpeak[3], locpeak[2], im.wcs, 0, 'wcs')
    print("imaging.point_source_find: Found peak at world coordinates %s" % str(sc))
    flux = im.data[:, :, locpeak[2], locpeak[3]]
    print("imaging.point_source_find: Flux is %s" % flux)
    # We also need the frequency values
    w = im.wcs.sub(['spectral'])
    frequency = w.wcs_pix2world(range(im.data.shape[0]), 1)
    return create_skycomponent(direction=sc, flux=flux, frequency=frequency, shape='point')


def find_flux_at_direction(im: Image, sc: SkyCoord, **kwargs):
    """ Find flux at a given direction, return SkyComponent

    :param im:
    :type Image:
    :param sc:
    :type SkyCoord:
    :returns: SkyComponent

    """
    print("imaging.find_flux_at_direction: Extracting flux at world coordinates %s" % str(sc))
    w = im.wcs.sub(['longitude', 'latitude'])
    pixloc = skycoord_to_pixel(sc, im.wcs, 0, 'wcs')
    print("imaging.find_flux_at_direction: Extracting flux at pixel coordinates %d %d" % (pixloc[0], pixloc[1]))
    flux = im.data[:, :, int(pixloc[1] + 0.5), int(pixloc[0] + 0.5)]
    print("imaging.find_flux_at_direction: Flux is %s" % flux)
    
    # We also need the frequency values
    w = im.wcs.sub(['spectral'])
    frequency = w.wcs_pix2world(range(im.data.shape[0]), 0)
    
    return create_skycomponent(direction=sc, flux=flux, frequency=frequency, shape='point')


class SkyModel:
    """ A skymodel consisting of a list of images and a list of components
    """
    # TODO: Fill out SkyModel

    def __init__(self):
        self.images = []
        self.components = []


def filter_skymodel(sm: SkyModel, **kwargs):
    """Filter the sky model

    :param sm:
    :type SkyModel:
    :param kwargs:
    :returns: SkyModel
    """
    print("SkyModel: No filter implemented yet")
    return sm


def add_skymodel(sm1: SkyModel, sm2: SkyModel):
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


def create_skymodel_from_image(im: Image):
    """ Create a skymodel from an image or image
    
    :param im:
    :type Image:
    :returns: SkyModel
    """
    sm = SkyModel()
    sm.images.append(im)
    return sm


def add_image_to_skymodel(sm: SkyModel, im: Image):
    """Add images to a sky model
    
    :param sm:
    :type SkyModel:
    :param im:
    :type Image:
    :returns: SkyModel
    """
    sm.images.append(im)
    return sm


def create_skymodel_from_component(comp: SkyComponent):
    """Create sky model from component
    
    :param comp:
    :type SkyComponent:
    :returns: SkyModel
    """
    sm = SkyModel()
    sm.components.append(comp)
    return sm


def add_component_to_skymodel(sm: SkyModel, comp: SkyComponent):
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

    flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [10.5, 0.0, 0.0, 0.0]])
    frequency = numpy.array([1.0e8, 1.5e8])
    direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
    comp = create_skycomponent(direction, flux, frequency, shape='Point', name="Mysource")

    kwargs = {}
    m31image = filter_skymodel(create_image_from_fits("./data/models/M31.MOD"), **kwargs)
    m31im = SkyModel()
    m31im.images.append(m31image)
    flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
    comp = create_skycomponent(direction, flux, frequency=numpy.arange(5e6, 300e6, 1e7), shape='Point',
                               name="Mysource")
    m31comp = SkyModel()
    m31comp.components.append(comp)
    m31added = add_skymodel(m31im, m31comp)
