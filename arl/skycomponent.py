# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

import numpy as numpy

from astropy.coordinates import SkyCoord

"""
Functions that represent and manipulate single components of sky brightness e.g. point source, Gaussian"""


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


if __name__ == '__main__':
    flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [10.5, 0.0, 0.0, 0.0]])
    frequency=numpy.array([1.0e8,1.5e8])
    direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
    comp = create_skycomponent(direction, flux, frequency, frequency='Point', shape='Point', name="Mysource")
