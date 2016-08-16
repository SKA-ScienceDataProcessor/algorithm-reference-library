# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

import numpy

from astropy.coordinates import SkyCoord

from arl.image import *

"""
Functions that represent and manipulate beam patterns

"""


class BeamPattern:
    """ An antenna or station BeamPattern
    
    A beam pattern contains the Mueller matrix as a function of direction, frequency, and polarisation.
    """
    
    # TODO: fill out BeamPattern
    def __init__(self):
        self.mueller = None
        self.wcs = None
        self.symmetry = None
        self.name = 'UNKNOWN'
        self.direction = None
        self.frequency = None
        self.params = None


def create_namedbeampattern(name: str = 'UNKNOWN', naxis: numpy.array = None, wcs: WCS = None, param: dict = None):
    """ A single BeamPattern

    :param type:
    :type str:
    :param direction:
    :type SkyCoord: for the calculation
    :param frequency:
    :type numpy.array:
    :returns: BeamPattern
    """
    sc = BeamPattern()
    self.mueller = None
    sc.name = name
    sc.direction = direction
    sc.frequency = frequency
    sc.params = param
    # We add the calculation of the actual Mueller matrices below
    print("creat_namedbeampattern: not yet implemented")
    return sc


def create_tabulatedbeampattern(name: str = 'UNKNOWN', bp: numpy.array = None, wcs: WCS = None, param: dict = None):
    """ A single BeamPattern created from an tabulated image

    :param name:
    :type str:
    :param bp:
    :type numpy.array:
    :param frequency:
    :type numpy.array:
    :returns: BeamPattern
    """
    sc = BeamPattern()
    sc.name = name
    sc.params = param
    # We add the application of the actual Mueller matrices below
    print("creat_tabulatedbeampattern: not yet implemented")
    return sc


def apply_beampattern(bp: BeamPattern, im: Image):
    """ Apply a beam pattern to an image

    """
    print("apply_beampattern: not yet implemented")
    return im


def extract_illuminationpattern(bp: BeamPattern, template: Image = None):
    """ Extract the illumination pattern for a given WCS

    """
    print("extract_illuminationpattern: not yet implemented")
    illum=Image(template.data, template.wcs, dtype='complex')
    return numpy.array()


if __name__ == '__main__':
    frequency = numpy.array([1.0e8, 1.5e8])
    param = {'diameter': 25.0, 'blockage': 1.8}
    bp = create_BeamPattern('BlockedAiry', direction=None, frequency=frequency, param=param)
