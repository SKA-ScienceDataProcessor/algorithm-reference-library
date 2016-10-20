# Tim Cornwell <realtimcornwell@gmail.com>
""" Data models used in ARL"""

from astropy.constants import c
from astropy.table import Table
import numpy


class Configuration:
    """ Describe a Configuration
    
    Has a Table with locations in x,y,z, and names, and overall location
    """
    
    def __init__(self, name='', data=None, location=None,
                 names="%s", xyz=None, mount="alt-az"):
        
        # Defaults
        if data is None and not xyz is None:
            nants = xyz.shape[0]
            if isinstance(names, str):
                names = [names % ant for ant in range(nants)]
            if isinstance(mount, str):
                mount = numpy.repeat(mount, nants)
            data = Table([names, xyz, mount],
                         names=['names', 'xyz', 'mount'])
        
        self.name = name
        self.data = data
        self.location = location
    
    @property
    def names(self):
        """ Names of the antennas/stations"""
        return self.data['names']
    
    @property
    def xyz(self):
        """ XYZ locations of antennas/stations
        """
        return self.data['xyz']
    
    @property
    def mount(self):
        """ Mount type
        """
        return self.data['mount']


class GainTable:
    """ Gain table with data: time, antenna, gain[:,chan,pol] columns
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
        """ Empty image
        """
        self.data = None
        self.wcs = None
    
    @property
    def nchan(self): return self.data.shape[0]
    
    @property
    def npol(self): return self.data.shape[1]
    
    @property
    def npixel(self): return self.data.shape[3]


class Skycomponent:
    """ A single Skycomponent with direction, flux, shape, and params for the shape
    
    """
    
    # TODO: fill out Skycomponent
    def __init__(self):
        """ Define the required structure
        """
        self.direction = None  # SkyCoord
        self.frequency = None  # numpy.array [nchan]
        self.name = None  # user friendly name
        self.flux = None  # numpy.array [nchan, npol]
        self.shape = None  # str e.g. 'Point' 'Gaussian'
        self.params = None  # numpy.array shape dependent parameters
    
    @property
    def nchan(self): return self.flux.shape[0]
    
    @property
    def npol(self): return self.flux.shape[1]


class Skymodel:
    """ A skymodel consisting of a list of images and a list of skycomponents
    
    """
    
    def __init__(self):
        self.images = []  # collection of numpy arrays
        self.components = []  # collection of SkyComponents


class Visibility:
    """ Visibility table class

    Visibility with uvw, time, a1, a2, vis, weight Columns in
    an astropy Table along with an attribute to hold the frequencies
    and an attribute to hold the direction.

    Visibility is defined to hold an observation with one set of frequencies and one
    direction.

    The data column has vis:[row,nchan,npol], uvw:[row,3]
    """
    
    def __init__(self, data=None, frequency=None, phasecentre=None, configuration=None,
                 uvw=None, time=None, antenna1=None, antenna2=None, vis=None, weight=None):
        if data is None and vis is not None:
            data = Table({'uvw': uvw, 'time': time,
                          'antenna1': antenna1, 'antenna2': antenna2,
                          'vis': vis, 'weight': weight
                          })
        
        self.data = data  # Astropy.table with columns uvw, time, a1, a2, vis, weight
        self.frequency = frequency  # numpy.array [nchan]
        self.phasecentre = phasecentre  # Phase centre of observation
        self.configuration = configuration  # Antenna/station configuration
    
    @property
    def nchan(self): return self.data['vis'].shape[1]
    
    @property
    def npol(self): return self.data['vis'].shape[2]
    
    @property
    def uvw(self): return self.data['uvw']
    
    @property
    def u(self):   return self.data['uvw'][:, 0]
    
    @property
    def v(self):   return self.data['uvw'][:, 1]
    
    @property
    def w(self):   return self.data['uvw'][:, 2]
    
    @property
    def time(self): return self.data['time']
    
    @property
    def antenna1(self): return self.data['antenna1']
    
    @property
    def antenna2(self): return self.data['antenna2']
    
    @property
    def vis(self): return self.data['vis']
    
    @property
    def weight(self): return self.data['weight']
    
    def uvw_lambda(self, channel=0):
        """ Calculates baseline coordinates in wavelengths. """
        return self.data['uvw'] * self.frequency[channel] / c.value


class QA:
    """ Quality assessment
    
    """
    
    # TODO: Implement some form of QA receipt and bundling
    
    def __init__(self, origin=None, data=None, context=None):
        self.origin = origin  # Name of function originating QA assessment
        self.data = data  # Dictionary containing standard fields
        self.context = context  # Context string (TBD)


def assert_same_chan_pol(o1, o2):
    """
    Assert that two entities indexed over channels and polarisations
    have the same number of them.
    """
    assert o1.npol == o2.npol, \
        "%s and %s have different number of polarisations: %d != %d" % \
        (type(o1).__name__, type(o2).__name__, o1.npol, o2.npol)
    assert o1.nchan == o2.nchan, \
        "%s and %s have different number of channels: %d != %d" % \
        (type(o1).__name__, type(o2).__name__, o1.nchan, o2.nchan)
