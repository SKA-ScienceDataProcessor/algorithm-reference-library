# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

import numpy as numpy

from astropy.coordinates import SkyCoord


class SkyComponent():
    """

    """

    def __init__(self, direction: SkyCoord, flux: numpy.array, shape: str = 'Point', name: str = 'Anon'):
        """
        A single SkyComponent with direction, flux, shape, and parameters for the shape
        :type name: str
        :param direction: SkyCoord
        :param flux: numpy.array[4]
        :param shape: 'Point' or 'Gaussian'
        :param name: str
        """
        self.direction = direction
        self.name = name
        self.flux = flux
        self.shape = shape
        self.params = None


if __name__ == '__main__':
    flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
    comp = SkyComponent(direction, flux, shape='Point', name="Mysource")
    print(dir(comp))
