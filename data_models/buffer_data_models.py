""" This is pending definition at the architecture level. For the moment, the memory data models are
repeated here.
"""

import logging
import sys
from typing import Union

from copy import copy, deepcopy

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame, ReceptorFrame

log = logging.getLogger(__name__)


class Buffer_Configuration:
    """ Describe a Configuration as locations in x,y,z, mount type, diameter, names, and
        overall location
    """
    
    def __init__(self, name='', data=None, location=None,
                 names="%s", xyz=None, mount="alt-az", frame=None,
                 receptor_frame=ReceptorFrame("linear"),
                 diameter=None):
        
        """Configuration object describing data for processing

        :param name:
        :param data:
        :param location:
        :param names:
        :param xyz:
        :param mount:
        :param frame:
        :param receptor_frame:
        :param diameter:
        """
        if data is None and xyz is not None:
            desc = [('names', '>U6'),
                    ('xyz', '>f8', (3,)),
                    ('diameter', '>f8'),
                    ('mount', '>U5')]
            nants = xyz.shape[0]
            if isinstance(names, str):
                names = [names % ant for ant in range(nants)]
            if isinstance(mount, str):
                mount = numpy.repeat(mount, nants)
            data = numpy.zeros(shape=[nants], dtype=desc)
            data['names'] = names
            data['xyz'] = xyz
            data['mount'] = mount
            data['diameter'] = diameter
        
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
    def diameter(self):
        """ diameter of antennas/stations
        """
        return self.data['diameter']
    
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


class Buffer_GainTable:
    """ Gain table with data_models: time, antenna, gain[:, chan, rec, rec], weight columns

    The weight is usually that output from gain solvers.
    """
    
    def __init__(self, data=None, gain: numpy.array = None, time: numpy.array = None, interval=None,
                 weight: numpy.array = None, residual: numpy.array = None, frequency: numpy.array = None,
                 receptor_frame: ReceptorFrame = ReceptorFrame("linear")):
        """ Create a gaintable from arrays

        The definition of gain is:

            Vobs = g_i g_j^* Vmodel

        :param interval:
        :param data:
        :param gain: [:, nchan, nrec, nrec]
        :param time: Centroid of solution
        :param interval: Interval of validity
        :param weight:
        :param residual:
        :param frequency:
        :param receptor_frame:
        :return: Gaintable
        """
        if data is None and gain is not None:
            nrec = receptor_frame.nrec
            nrows = gain.shape[0]
            nants = gain.shape[1]
            nchan = gain.shape[2]
            assert len(frequency) == nchan, "Discrepancy in frequency channels"
            desc = [('gain', '>c16', (nants, nchan, nrec, nrec)),
                    ('weight', '>f8', (nants, nchan, nrec, nrec)),
                    ('residual', '>f8', (nchan, nrec, nrec)),
                    ('time', '>f8'),
                    ('interval', '>f8')]
            data = numpy.zeros(shape=[nrows], dtype=desc)
            data['gain'] = gain
            data['weight'] = weight
            data['time'] = time
            data['interval'] = interval
            data['residual'] = residual
            
        self.data = data
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
    def interval(self):
        return self.data['interval']

    @property
    def gain(self):
        return self.data['gain']
    
    @property
    def weight(self):
        return self.data['weight']
    
    @property
    def residual(self):
        return self.data['residual']

    @property
    def ntimes(self):
        return self.data['gain'].shape[0]

    @property
    def nants(self):
        return self.data['gain'].shape[1]

    @property
    def nchan(self):
        return self.data['gain'].shape[2]
    
    @property
    def nrec(self):
        return self.receptor_frame.nrec

    def __str__(self):
        """Default printer for GainTable

        """
        s = "GainTable:\n"
        s += "\tTimes: %s\n" % str(self.ntimes)
        s += "\tData shape: %s\n" % str(self.data.shape)
        s += "\tReceptor frame: %s\n" % str(self.receptor_frame.type)
        return s


class Buffer_Image:
    """Image class with Image data (as a numpy.array) and the AstroPy `implementation of
    a World Coodinate System <http://docs.astropy.org/en/stable/wcs>`_

    Many operations can be done conveniently using numpy libs on Image.data_models.

    Most of the imaging libs require an image in canonical format:
    - 4 axes: RA, DEC, POL, FREQ

    The conventions for indexing in WCS and numpy are opposite.
    - In astropy.wcs, the order is (longitude, latitude, polarisation, frequency)
    - in numpy, the order is (frequency, polarisation, latitude, longitude)

    .. warning::
        The polarisation_frame is kept in two places, the WCS and the polarisation_frame
        variable. The latter should be considered definitive.

    """
    
    def __init__(self):
        """ Empty image
        """
        self.data = None
        self.wcs = None
        self.polarisation_frame = None
    
    def size(self):
        """ Return size in GB
        """
        size = 0
        size += self.data.nbytes
        return size / 1024.0 / 1024.0 / 1024.0

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    @property
    def nchan(self):
        return self.data.shape[0]
    
    @property
    def npol(self):
        return self.data.shape[1]
    
    @property
    def nheight(self):
        return self.data.shape[2]
    
    @property
    def nwidth(self):
        return self.data.shape[3]
    
    @property
    def frequency(self):
        w = self.wcs.sub(['spectral'])
        return w.wcs_pix2world(range(self.nchan), 0)[0]
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def phasecentre(self):
        return SkyCoord(self.wcs.wcs.crval[0] * u.deg, self.wcs.wcs.crval[1] * u.deg)
    
    def __str__(self):
        """Default printer for Image

        """
        s = "Image:\n"
        s += "\tShape: %s\n" % str(self.data.shape)
        s += "\tWCS: %s\n" % self.wcs
        s += "\tPolarisation frame: %s\n" % str(self.polarisation_frame.type)
        return s


class Buffer_Skycomponent:
    """Skycomponents are used to represent compact sources on the sky. They possess direction,
    flux as a function of frequency and polarisation, shape (with params), and polarisation frame.

    For example, the following creates and predicts the visibility from a collection of point sources
    drawn from the GLEAM catalog::

        sc = create_low_test_skycomponents_from_gleam(flux_limit=1.0,
                                                    polarisation_frame=PolarisationFrame("stokesIQUV"),
                                                    frequency=frequency, kind='cubic',
                                                    phasecentre=phasecentre,
                                                    radius=0.1)
        model = create_image_from_visibility(vis, cellsize=0.001, npixel=512, frequency=frequency,
                                            polarisation_frame=PolarisationFrame('stokesIQUV'))

        bm = create_low_test_beam(model=model)
        sc = apply_beam_to_skycomponent(sc, bm)
        vis = predict_skycomponent_visibility(vis, sc)
    """
    
    def __init__(self,
                 direction=None, frequency=None, name=None, flux=None, shape='Point',
                 polarisation_frame=PolarisationFrame('stokesIQUV'), params={}):
        """ Define the required structure

        :param direction: SkyCoord
        :param frequency: numpy.array [nchan]
        :param name: user friendly name
        :param flux: numpy.array [nchan, npol]
        :param shape: str e.g. 'Point' 'Gaussian'
        :param params: numpy.array shape dependent parameters
        :param polarisation_frame: Polarisation_frame
        """
        
        self.direction = direction
        self.frequency = numpy.array(frequency)
        self.name = name
        self.flux = numpy.array(flux)
        self.shape = shape
        self.params = params
        self.polarisation_frame = polarisation_frame
        
        assert len(self.frequency.shape) == 1, frequency
        assert len(self.flux.shape) == 2, flux
        assert self.frequency.shape[0] == self.flux.shape[0], \
            "Frequency shape %s, flux shape %s" % (self.frequency.shape, self.flux.shape)
        assert polarisation_frame.npol == self.flux.shape[1], \
            "Polarisation is %s, flux shape %s" % (polarisation_frame.type, self.flux.shape)
    
    @property
    def nchan(self):
        return self.flux.shape[0]
    
    @property
    def npol(self):
        return self.flux.shape[1]
    
    def __str__(self):
        """Default printer for Skycomponent

        """
        s = "Skycomponent:\n"
        s += "\tName: %s\n" % self.name
        s += "\tFlux: %s\n" % self.flux
        s += "\tFrequency: %s\n" % self.frequency
        s += "\tDirection: %s\n" % self.direction
        s += "\tShape: %s\n" % self.shape

        class SkyModel:
            """ A model for the sky
            """
    
            def __init__(self, images=[], components=[], fixed=False):
                """ A model of the sky as a list of images and a list of components

                """
                self.images = [im for im in images]
                self.components = [sc for sc in components]
                self.fixed = fixed
    
            def __str__(self):
                """Default printer for SkyModel

                """
                s = "SkyModel: fixed: %s\n" % self.fixed
                for i, sc in enumerate(self.components):
                    s += str(sc)
                s += "\n"
        
                for i, im in enumerate(self.images):
                    s += str(im)
                s += "\n"
        
                return s

        s += "\tParams: %s\n" % self.params
        s += "\tPolarisation frame: %s\n" % str(self.polarisation_frame.type)
        return s


class Buffer_SkyModel:
    """ A model for the sky
    """
    
    def __init__(self, images=[], components=[], fixed=False):
        """ A model of the sky as a list of images and a list of components
        
        Use copy_skymodel to make a proper copy of skymodel

        """
        self.images = [im for im in images]
        self.components = [sc for sc in components]
        self.fixed = fixed
    
    def __str__(self):
        """Default printer for SkyModel

        """
        s = "SkyModel: fixed: %s\n" % self.fixed
        for i, sc in enumerate(self.components):
            s += str(sc)
        s += "\n"
        
        for i, im in enumerate(self.images):
            s += str(im)
        s += "\n"
        
        return s


class Buffer_Visibility:
    """ Visibility table class

    Visibility with uvw, time, integration_time, frequency, channel_bandwidth, a1, a2, vis, weight
    as separate columns in a numpy structured array, The fundemental unit is a complex vector of polarisation.

    Visibility is defined to hold an observation with one direction.
    Polarisation frame is the same for the entire data set and can be stokes, circular, linear
    The configuration is also an attribute

    The phasecentre is the direct of delay tracking i.e. n=0. If uvw are rotated then this
    should be updated with the new delay tracking centre. This is important for wstack and wproject
    algorithms.

    If a visibility is created by coalescence then the cindex column is filled with a pointer to the
    row in the original block visibility that this row has a value for. The original blockvisibility
    is also preserves as n attribute so that decoalescence is expedited. If you don't need that then
    the storage can be released by setting self.blockvis to None
    """
    
    def __init__(self,
                 data=None, frequency=None, channel_bandwidth=None,
                 phasecentre=None, configuration=None, uvw=None,
                 time=None, antenna1=None, antenna2=None, vis=None,
                 weight=None, imaging_weight=None, integration_time=None,
                 polarisation_frame=PolarisationFrame('stokesI'), cindex=None,
                 blockvis=None):
        """Visibility

        :param data:
        :param frequency:
        :param channel_bandwidth:
        :param phasecentre:
        :param configuration:
        :param uvw:
        :param time:
        :param antenna1:
        :param antenna2:
        :param vis:
        :param weight:
        :param imaging_weight:
        :param integration_time:
        :param polarisation_frame:
        :param cindex:
        :param blockvis:
        """
        if data is None and vis is not None:
            if imaging_weight is None:
                imaging_weight = weight
            nvis = vis.shape[0]
            assert len(time) == nvis
            assert len(frequency) == nvis
            assert len(channel_bandwidth) == nvis
            assert len(antenna1) == nvis
            assert len(antenna2) == nvis
            
            npol = polarisation_frame.npol
            desc = [('index', '>i8'),
                    ('uvw', '>f8', (3,)),
                    ('time', '>f8'),
                    ('frequency', '>f8'),
                    ('channel_bandwidth', '>f8'),
                    ('integration_time', '>f8'),
                    ('antenna1', '>i8'),
                    ('antenna2', '>i8'),
                    ('vis', '>c16', (npol,)),
                    ('weight', '>f8', (npol,)),
                    ('imaging_weight', '>f8', (npol,))]
            data = numpy.zeros(shape=[nvis], dtype=desc)
            data['index'] = list(range(nvis))
            data['uvw'] = uvw
            data['time'] = time
            data['frequency'] = frequency
            data['channel_bandwidth'] = channel_bandwidth
            data['integration_time'] = integration_time
            data['antenna1'] = antenna1
            data['antenna2'] = antenna2
            data['vis'] = vis
            data['weight'] = weight
            data['imaging_weight'] = imaging_weight
        
        self.data = data  # numpy structured array
        self.cindex = cindex
        self.blockvis = blockvis
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
    def index(self):
        return self.data['index']
    
    @property
    def npol(self):
        return self.polarisation_frame.npol
    
    @property
    def nvis(self):
        return self.data['vis'].shape[0]
    
    @property
    def uvw(self):  # In wavelengths in Visibility
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


class Buffer_BlockVisibility:
    """ Block Visibility table class

    BlockVisibility with uvw, time, integration_time, frequency, channel_bandwidth, pol,
    a1, a2, vis, weight Columns in a numpy structured array.
    
    BlockVisibility is defined to hold an observation with one direction.

    The phasecentre is the direct of delay tracking i.e. n=0. If uvw are rotated then this
    should be updated with the new delay tracking centre. This is important for wstack and wproject
    algorithms.

    Polarisation frame is the same for the entire data set and can be stokesI, circular, linear
    
    The configuration is also an attribute
    """
    
    def __init__(self,
                 data=None, frequency=None, channel_bandwidth=None,
                 phasecentre=None, configuration=None, uvw=None,
                 time=None, vis=None, weight=None, integration_time=None,
                 polarisation_frame=PolarisationFrame('stokesI')):
        """BlockVisibility

        :param data:
        :param frequency:
        :param channel_bandwidth:
        :param phasecentre:
        :param configuration:
        :param uvw:
        :param time:
        :param vis:
        :param weight:
        :param integration_time:
        :param polarisation_frame:
        """
        if data is None and vis is not None:
            ntimes, nants, _, nchan, npol = vis.shape
            assert vis.shape == weight.shape
            assert len(frequency) == nchan
            assert len(channel_bandwidth) == nchan
            desc = [('index', '>i8'),
                    ('uvw', '>f8', (nants, nants, 3)),
                    ('time', '>f8'),
                    ('integration_time', '>f8'),
                    ('vis', '>c16', (nants, nants, nchan, npol)),
                    ('weight', '>f8', (nants, nants, nchan, npol))]
            data = numpy.zeros(shape=[ntimes], dtype=desc)
            data['index'] = list(range(ntimes))
            data['uvw'] = uvw
            data['time'] = time
            data['integration_time'] = integration_time
            data['vis'] = vis
            data['weight'] = weight
        
        self.data = data  # numpy structured array
        self.frequency = frequency
        self.channel_bandwidth = channel_bandwidth
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
    def time(self):
        return self.data['time']
    
    @property
    def integration_time(self):
        return self.data['integration_time']
    
    @property
    def nvis(self):
        return self.data.size


class Buffer_QA:
    """ Quality assessment

    """
    
    def __init__(self, origin=None, data=None, context=None):
        """QA

        :param origin:
        :param data:
        :param context:
        """
        self.origin = origin  # Name of function originating QA assessment
        self.data = data  # Dictionary containing standard fields
        self.context = context  # Context string
    
    def __str__(self):
        """Default printer for QA

        """
        s = "Quality assessment:\n"
        s += "\tOrigin: %s\n" % self.origin
        s += "\tContext: %s\n" % self.context
        s += "\tData:\n"
        for dataname in self.data.keys():
            s += "\t\t%s: %r\n" % (dataname, str(self.data[dataname]))
        return s


class Buffer_Science_Data_model:
    """ Buffered version of Science Data Model"""
    
    def __init__(self):
        pass
    
    def __str__(self):
        """ Deflaut printer for Science Data Model

        :return:
        """
        return ""

