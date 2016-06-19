# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

from collections import namedtuple
import numpy as numpy

from astropy.coordinates import SkyCoord
import astropy.units as u

from functions.fimage import *


def fcomp():
    """ Component with SkyCoord, numpy.array
    """
    fc=namedtuple('fcomp', ['direction', 'name', 'flux', 'shape', 'params'])
    fc.direction = SkyCoord(0.0 * u.rad, 0.0 * u.rad),
    fc.name = str(""),
    fc.flux = numpy.array([0.0,0.0,0.0,0.0]),
    fc.shape = 'Point',
    fc.params = None
    return fc

def fcomp_construct(direction: SkyCoord, flux: numpy.array, shape: str = 'Point', name: str = 'Anon'):
    """
    A single component with direcction, flux, shape, and parameters for the shape
    :type name: str
    :param dir: SkyCoord
    :param flux: numpy.array[4]
    :param shape: 'Point' or 'Gaussian'
    :param params:
    """
    fc=fcomp()
    fc.direction = direction
    fc.name = name
    fc.flux = flux
    fc.shape = shape
    fc.params = None
    return fc


if __name__ == '__main__':
    flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
    comp = fcomp_construct(direction, flux, shape='Point', name="Mysource")
    print(dir(comp))
