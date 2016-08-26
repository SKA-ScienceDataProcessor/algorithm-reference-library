# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

import numpy
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord

class Configuration:
    """ Describe a Configuration
    """
    
    def __init__(self):
        self.name = ''
        self.data = None
        self.location = None


class GainTable:
    """
    Gain table with time, antenna, gain[:,chan,pol] columns
    """
    
    # TODO: Implement gaintables with Jones and Mueller matrices
    
    def __init__(self):
        self.data = None
        self.frequency = None


class Image:
    """Image class with Image data (as a numpy.array) and optionally the AstroPy WCS.

    Many operations can be done conveniently using numpy arl on Image.data.

    Most of the imaging arl require an image in canonical format:
    - 4 axes: RA, DEC, POL, FREQ

    The conventions for indexing in WCS and numpy are opposite.
    - In astropy.wcs, the order is (longitude, latitude, polarisation, frequency)
    - in numpy, the order is (frequency, polarisation, latitude, longitude)

    """
    
    def __init__(self):
        self.data = None
        self.wcs = None


class SkyComponent:
    """ A single SkyComponent with direction, flux, shape, and parameters for the shape
    
    """
    # TODO: fill out SkyComponent
    def __init__(self):
        self.direction = None
        self.frequency = None
        self.name = None
        self.flux = None
        self.shape = None
        self.params = None


class SkyModel:
    """ A skymodel consisting of a list of images and a list of skycomponents
    
    """
    # TODO: Fill out SkyModel

    def __init__(self):
        self.images = []
        self.components = []


class Visibility:
    """ Visibility table class

    Visibility with uvw, time, a1, a2, vis, weight Columns in
    an astropy Table along with an attribute to hold the frequencies
    and an attribute to hold the direction.

    Visibility is defined to hold an observation with one set of frequencies and one
    direction.

    The data column has vis:[row,nchan,npol], uvw:[row,3]
    """
    
    def __init__(self):
        self.data = None
        self.frequency = None
        self.phasecentre = None
        self.configuration = None


class AQ:
    """
    Quality assessment
    """
    
    # TODO: Implement some form of QA receipt and bundling
    
    def __init__(self):
        self.data = None
        self.context = None