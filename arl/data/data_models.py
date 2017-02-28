# Tim Cornwell <realtimcornwell@gmail.com>
""" Data models used in ARL"""

import logging
import sys

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.data.polarisation import Polarisation_Frame, Receptor_Frame

log = logging.getLogger(__name__)


class Configuration:
    """ Describe a Configuration
    
    Has a Table with locations in x,y,z, and names, and overall location
    """
    
    def __init__(self, name='', data=None, location=None,
                 names="%s", xyz=None, mount="alt-az", frame=None, receptor_frame=Receptor_Frame("linear")):
        
        # Defaults
        if data is None and not xyz is None:
            desc = [('names', '<U6'),
                    ('xyz', '<f8', (3,)),
                    ('mount', '<U5')]
            nants = xyz.shape[0]
            if isinstance(names, str):
                names = [names % ant for ant in range(nants)]
            if isinstance(mount, str):
                mount = numpy.repeat(mount, nants)
            data = numpy.zeros(shape=[nants], dtype=desc)
            data['names'] = names
            data['xyz'] = xyz
            data['mount'] = mount
        
        self.name = name
        self.data = data
        self.location = location
        self.frame = frame
        self.receptor_frame = receptor_frame
    
    def size(self):
        """ Return size in GB
        """
        size = 0
        size += self.data.size * sys.getsizeof(self.data)
        return size / 1024.0 / 1024.0 / 1024.0
    
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
    """ Gain table with data: time, antenna, gain[:,chan,pol], weight columns
    
    The weight is usually that output from gain solvers.
    """
    
    def __init__(self, data=None, gain: numpy.array = None, time: numpy.array = None, antenna: numpy.array = None,
                 weight: numpy.array = None, frequency: numpy.array = None,
                 receptor_frame: Receptor_Frame = Receptor_Frame("linear")):
        """ Create a gaintable from arrays
        
        The definition of gain is:
            Vobs = g_i g_j^* Vmodel

        :param gain: [npol, nchan]
        :param time:
        :param antenna:
        :param weight:
        :param frequency:
        :returns: Gaintable
        """
        if data is None and not gain is None:
            npol = receptor_frame.npol
            nrows = gain.shape[0]
            nants = gain.shape[1]
            nchan = gain.shape[2]
            assert len(frequency) == nchan, "Discrepancy in frequency channels"
            desc = [('gain', '<c16', (nants, nchan, npol)),
                    ('weight', '<f8', (nants, nchan, npol)),
                    ('time', '<f8')]
            self.data = numpy.zeros(shape=[nrows], dtype=desc)
            self.data['gain'] = gain
            self.data['weight'] = weight
            self.data['time'] = time
        self.frequency = frequency
        self.receptor_frame = receptor_frame
    
    def size(self):
        """ Return size in GB
        """
        size = 0
        size += self.data.size * sys.getsizeof(self.data)
        return size / 1024.0 / 1024.0 / 1024.0

    @property
    def time(self):
        return self.data['time']
    
    @property
    def gain(self):
        return self.data['gain']

    @property
    def weight(self):
        return self.data['weight']

    @property
    def nants(self):
        return self.data['gain'].shape[1]



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
    
    def size(self):
        """ Return size in GB
        """
        size = 0
        size += self.data.nbytes
        return size / 1024.0 / 1024.0 / 1024.0
    
    @property
    def nchan(self): return self.data.shape[0]
    
    @property
    def npol(self): return self.data.shape[1]

    @property
    def nheight(self): return self.data.shape[2]

    @property
    def nwidth(self): return self.data.shape[3]
    
    @property
    def frequency(self):
        w = self.wcs.sub(['spectral'])
        return w.wcs_pix2world(range(self.nchan), 1)[0]
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def phasecentre(self): return SkyCoord(self.wcs.wcs.crval[0] * u.deg, self.wcs.wcs.crval[1] * u.deg)
    

class Skycomponent:
    """ A single Skycomponent with direction, flux, shape, and params for the shape
    
    """
    
    def __init__(self,
                 direction=None, frequency=None, name=None, flux=None, shape='Point', **kwargs):
        """ Define the required structure

        :param direction: SkyCoord
        :param frequency: numpy.array [nchan]
        :param name: user friendly name
        :param flux: numpy.array [nchan, npol]
        :param shape: str e.g. 'Point' 'Gaussian'
        :param params: numpy.array shape dependent parameters
        """
        
        self.direction = direction
        self.frequency = numpy.array(frequency)
        self.name = name
        self.flux = numpy.array(flux)
        self.shape = shape
        self.params = kwargs
        
        assert len(self.frequency.shape) == 1
        assert len(self.flux.shape) == 2
        assert self.frequency.shape[0] == self.flux.shape[0], "Frequency shape %s, flux shape %s" % (
            self.frequency.shape, self.flux.shape)
    
    @property
    def nchan(self): return self.flux.shape[0]
    
    @property
    def npol(self): return self.flux.shape[1]
    
    def __str__(self):
        """Default printer for Skycomponent

        """
        s = "Skycomponent:\n"
        s += "\tFlux: %s\n" % (self.flux)
        s += "\tDirection: %s\n" % (self.direction)
        s += "\tShape: %s\n" % (self.shape)
        s += "\tParams: %s\n" % (self.params)
        return s


class Skymodel:
    """ A skymodel consisting of a list of images and a list of skycomponents
    
    """
    
    def __init__(self):
        self.images = []  # collection of numpy arrays
        self.components = []  # collection of SkyComponents


class Visibility:
    """ Visibility table class

    Visibility with uvw, time, integration_time, frequency, channel_bandwidth, pol, a1, a2, vis, weight Columns in
    a numpy structured array
    Visibility is defined to hold an observation with one direction.
    Polarisation frame is the same for the entire data set and can be stokes, circular, linear
    The configuration is also an attribute
    """
    
    def __init__(self,
                 data=None, frequency=None, channel_bandwidth=None, phasecentre=None, configuration=None,
                 uvw=None, time=None, antenna1=None, antenna2=None, polarisation=None,
                 vis=None, weight=None, imaging_weight=None, integration_time=None,
                 polarisation_frame = Polarisation_Frame('stokesI')):
        if data is None and vis is not None:
            if imaging_weight is None:
                imaging_weight = weight
            nvis = vis.shape[0]
            desc = [('uvw', '<f8', (3,)),
                    ('time', '<f8'),
                    ('frequency', '<f8'),
                    ('channel_bandwidth', '<f8'),
                    ('polarisation', '<i8'),
                    ('integration_time', '<f8'),
                    ('antenna1', '<i8'),
                    ('antenna2', '<i8'),
                    ('vis', '<c16'),
                    ('weight', '<f8'),
                    ('imaging_weight', '<f8')]
            data = numpy.zeros(shape=[nvis], dtype=desc)
            data['uvw'] = uvw
            data['time'] = time
            data['frequency'] = frequency
            data['channel_bandwidth'] = channel_bandwidth
            data['polarisation'] = polarisation
            data['integration_time'] = integration_time
            data['antenna1'] = antenna1
            data['antenna2'] = antenna2
            data['vis'] = vis
            data['weight'] = weight
            data['imaging_weight'] = imaging_weight
            
        self.data = data  # numpy structured array
        self.phasecentre = phasecentre  # Phase centre of observation
        self.configuration = configuration  # Antenna/station configuration
        self.polarisation_frame = polarisation_frame
    
    def size(self):
        """ Return size in GB
        """
        size = 0
        for col in self.data.dtype.fields.keys():
            size += self.data[col].nbytes
        return size / 1024.0 / 1024.0 / 1024.0
    
    @property
    def nvis(self):
        return self.data['vis'].shape[0]
    
    @property
    def uvw(self): # In wavelengths in Visibility
        return self.data['uvw']
    
    @property
    def u(self):
        return self.data['uvw'][:, 0]
    
    @property
    def v(self):
        return self.data['uvw'][:, 1]
    
    @property
    def w(self):
        return self.data['uvw'][:, 2]

    @property
    def time(self):
        return self.data['time']

    @property
    def integration_time(self):
        return self.data['integration_time']

    @property
    def frequency(self):
        return self.data['frequency']

    @property
    def channel_bandwidth(self):
        return self.data['channel_bandwidth']

    @property
    def polarisation(self):
        return self.data['polarisation']

    @property
    def antenna1(self):
        return self.data['antenna1']
    
    @property
    def antenna2(self):
        return self.data['antenna2']
    
    @property
    def vis(self):
        return self.data['vis']
    
    @property
    def weight(self):
        return self.data['weight']
    
    @property
    def imaging_weight(self):
        return self.data['imaging_weight']


class BlockVisibility:
    """ Block Visibility table class
    
    Visibility with uvw, time, integration_time, frequency, channel_bandwidth, pol, a1, a2, vis, weight Columns in
    a numpy structured array
    Visibility is defined to hold an observation with one direction.
    Polarisation frame is the same for the entire data set and can be stokes, circular, linear
    The configuration is also an attribute
    """
    
    def __init__(self,
                 data=None, frequency=None, channel_bandwidth=None, phasecentre=None, configuration=None,
                 uvw=None, time=None,
                 vis=None, weight=None, integration_time=None,
                 polarisation_frame=Polarisation_Frame('stokesI')):
        if data is None and vis is not None:
            ntimes = len(time)
            assert vis.shape[0] == ntimes
            nants = vis.shape[1]
            assert vis.shape[2] == nants
            nchan = vis.shape[3]
            npol = vis.shape[4]
            desc = [('uvw', '<f8', (nants, nants, 3,)),
                    ('time', '<f8'),
                    ('integration_time', '<f8'),
                    ('vis', '<c16', (nants, nants, nchan, npol)),
                    ('weight', '<f8', (nants, nants, nchan, npol))]
            data = numpy.zeros(shape=[ntimes], dtype=desc)
            data['uvw'] = uvw
            data['time'] = time
            data['integration_time'] = integration_time
            data['vis'] = vis
            data['weight'] = weight
        
        self.data = data  # numpy structured array
        self.time = time
        self.frequency = frequency
        self.phasecentre = phasecentre  # Phase centre of observation
        self.configuration = configuration  # Antenna/station configuration
        self.polarisation_frame = polarisation_frame
    
    def size(self):
        """ Return size in GB
        """
        size = 0
        for col in self.data.dtype.fields.keys():
            size += self.data[col].nbytes
        return size / 1024.0 / 1024.0 / 1024.0

    @property
    def nchan(self):
        return self.data['vis'].shape[3]

    @property
    def npol(self):
        return self.data['vis'].shape[4]

    @property
    def nants(self):
        return self.data['vis'].shape[1]
    
    @property
    def uvw(self):  # In wavelengths meters
        return self.data['uvw']
    
    @property
    def u(self):
        return self.data['uvw'][:, 0]
    
    @property
    def v(self):
        return self.data['uvw'][:, 1]
    
    @property
    def w(self):
        return self.data['uvw'][:, 2]
    
    @property
    def vis(self):
        return self.data['vis']

    @property
    def weight(self):
        return self.data['weight']

    @property
    def integration_time(self):
        return self.data['integration_time']

    @property
    def nvis(self):
        return self.data.size


class QA:
    """ Quality assessment
    
    """
    
    def __init__(self, origin=None, data=None, context=None):
        self.origin = origin  # Name of function originating QA assessment
        self.data = data  # Dictionary containing standard fields
        self.context = context  # Context string (TBD)
    
    def __str__(self):
        """Default printer for QA
        
        """
        s = "Quality assessment:\n"
        s += "\tOrigin: %s\n" % (self.origin)
        s += "\tContext: %s\n" % (self.context)
        s += "\tData:\n"
        for dataname in self.data.keys():
            s += "\t\t%s: %s\n" % (dataname, str(self.data[dataname]))
        return s


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
