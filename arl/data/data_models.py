"""The data models used in ARL are:

.. image:: ./ARL_data.png
   :scale: 75 %

The principle transitions between the data models:

.. image:: ./ARL_transitions.png
   :scale: 75 %

.. note::
    There are two visibility formats:

    :class:`BlockVisibility` is conceived as an ingest and calibration format. The visibility data are kept in a block
    of shape (number antennas, number antennas, number channels, number polarisation). One block is kept per integration. The other columns are time and uvw. The sampling in time is therefore the same for all baselines.

    :class:`Visibility` is designed to hold coalesced data where the integration time and channel width can vary with baseline length. The visibility data are kept in a visibility vector of length equal to the number of polarisations. Everything else is a separate column: time, frequency, uvw, channel_bandwidth, integration time.


"""

import logging
import sys
from typing import Union

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.data.polarisation import PolarisationFrame, ReceptorFrame

log = logging.getLogger(__name__)


class Configuration:
    """ Describe a Configuration as locations in x,y,z, mount type, diameter, names, and overall location
    """
    
    def __init__(self, name='', data=None, location=None,
                 names="%s", xyz=None, mount="alt-az", frame=None, receptor_frame=ReceptorFrame("linear"),
                 diameter=None):
        
        """

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


class GainTable:
    """ Gain table with data: time, antenna, gain[:, chan, rec, rec], weight columns
    
    The weight is usually that output from gain solvers.
    """
    
    def __init__(self, data=None, gain: numpy.array = None, time: numpy.array = None, weight: numpy.array = None,
                 residual: numpy.array = None, frequency: numpy.array = None,
                 receptor_frame: ReceptorFrame = ReceptorFrame("linear")):
        """ Create a gaintable from arrays
        
        The definition of gain is:
        
            Vobs = g_i g_j^* Vmodel

        :param data:
        :param gain: [:, nchan, nrec, nrec]
        :param time:
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
                    ('time', '>f8')]
            self.data = numpy.zeros(shape=[nrows], dtype=desc)
            self.data['gain'] = gain
            self.data['weight'] = weight
            self.data['time'] = time
            self.data['residual'] = residual
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
    def residual(self):
        return self.data['residual']
    
    @property
    def nants(self):
        return self.data['gain'].shape[1]
    
    @property
    def nchan(self):
        return self.data['gain'].shape[2]
    
    @property
    def nrec(self):
        return self.receptor_frame.nrec


class Image:
    """Image class with Image data (as a numpy.array) and the AstroPy `implementation of
    a World Coodinate System <http://docs.astropy.org/en/stable/wcs>`_

    Many operations can be done conveniently using numpy arl on Image.data.

    Most of the imaging arl require an image in canonical format:
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
    """Skycomponents are used to represent compact sources on the sky. They possess direction, flux as a function of
    frequency and polarisation, shape (with params), and polarisation frame.
    
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
        vis = predict_skycomponent_blockvisibility(vis, sc)
    """
    
    def __init__(self,
                 direction=None, frequency=None, name=None, flux=None, shape='Point',
                 polarisation_frame=PolarisationFrame('stokesIQUV'), **kwargs):
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
        self.params = kwargs
        self.polarisation_frame = polarisation_frame
        
        assert len(self.frequency.shape) == 1
        assert len(self.flux.shape) == 2
        assert self.frequency.shape[0] == self.flux.shape[0], "Frequency shape %s, flux shape %s" % (
            self.frequency.shape, self.flux.shape)
        assert polarisation_frame.npol == self.flux.shape[1], "Polarisation is %s, flux shape %s" % (
            polarisation_frame.type, self.flux.shape)
    
    @property
    def nchan(self): return self.flux.shape[0]
    
    @property
    def npol(self): return self.flux.shape[1]
    
    def __str__(self):
        """Default printer for Skycomponent

        """
        s = "Skycomponent:\n"
        s += "\tFlux: %s\n" % self.flux
        s += "\tDirection: %s\n" % self.direction
        s += "\tShape: %s\n" % self.shape
        s += "\tParams: %s\n" % self.params
        s += "\tPolarisation frame %s\n" % self.polarisation_frame
        return s


class Visibility:
    """ Visibility table class

    Visibility with uvw, time, integration_time, frequency, channel_bandwidth, a1, a2, vis, weight Columns in
    a numpy structured array, The fundemental unit is a complex vector of polarisation.
    
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
                 data=None, frequency=None, channel_bandwidth=None, phasecentre=None, configuration=None,
                 uvw=None, time=None, antenna1=None, antenna2=None, vis=None, weight=None, imaging_weight=None,
                 integration_time=None, polarisation_frame=PolarisationFrame('stokesI'), cindex=None,
                 blockvis=None):
        """

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
            desc = [('uvw', '>f8', (3,)),
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


class BlockVisibility:
    """ Block Visibility table class
    
    Visibility with uvw, time, integration_time, frequency, channel_bandwidth, pol, a1, a2, vis, weight Columns in
    a numpy structured array
    Visibility is defined to hold an observation with one direction.

    The phasecentre is the direct of delay tracking i.e. n=0. If uvw are rotated then this
    should be updated with the new delay tracking centre. This is important for wstack and wproject
    algorithms.
    
    Polarisation frame is the same for the entire data set and can be stokes, circular, linear
    The configuration is also an attribute
    """
    
    def __init__(self,
                 data=None, frequency=None, channel_bandwidth=None, phasecentre=None, configuration=None,
                 uvw=None, time=None, vis=None, weight=None, integration_time=None,
                 polarisation_frame=PolarisationFrame('stokesI')):
        """

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
            desc = [('uvw', '>f8', (nants, nants, 3)),
                    ('time', '>f8'),
                    ('integration_time', '>f8'),
                    ('vis', '>c16', (nants, nants, nchan, npol)),
                    ('weight', '>f8', (nants, nants, nchan, npol))]
            data = numpy.zeros(shape=[ntimes], dtype=desc)
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


class QA:
    """ Quality assessment
    
    """
    
    def __init__(self, origin=None, data=None, context=None):
        """

        :param origin:
        :param data:
        :param context:
        """
        self.origin = origin  # Name of function originating QA assessment
        self.data = data  # Dictionary containing standard fields
        self.context = context  # Context string (TBD)
    
    def __str__(self):
        """Default printer for QA
        
        """
        s = "Quality assessment:\n"
        s += "\tOrigin: %s\n" % self.origin
        s += "\tContext: %s\n" % self.context
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


def assert_vis_gt_compatible(vis: Union[Visibility, BlockVisibility], gt: GainTable):
    """ Check if visibility and gaintable are compatible
    
    :param vis:
    :param gt:
    :return:
    """
    assert vis.nchan == gt.nchan
    assert vis.npol == gt.nrec * gt.nrec
