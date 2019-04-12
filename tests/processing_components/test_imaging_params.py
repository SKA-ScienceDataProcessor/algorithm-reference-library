""" Unit tests for Fourier transform processor params


"""
import logging
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from processing_components.imaging.base import create_image_from_visibility
from processing_components.simulation.testing_support import create_low_test_image_from_gleam
from processing_components.simulation.configurations import create_named_configuration
from processing_components.visibility.base import create_visibility
from processing_library.imaging.imaging_params import get_frequency_map

log = logging.getLogger(__name__)


class TestImagingParams(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.vnchan = 7
        self.lowcore = create_named_configuration('LOWBD2', rmax=300.0)
        self.times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 7)
        self.frequency = numpy.linspace(8e7, 1.2e8, self.vnchan)
        self.startfrequency = numpy.array([8e7])
        self.channel_bandwidth = numpy.array(self.vnchan * [(1.0 - 1.0e-7) * (self.frequency[1] - self.frequency[0])])
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_visibility(self.lowcore, times=self.times, frequency=self.frequency,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('stokesI'),
                                     channel_bandwidth=self.channel_bandwidth)
        self.model = create_image_from_visibility(self.vis, npixel=128, cellsize=0.001, nchan=self.vnchan,
                                                  frequency=self.startfrequency)
    
    def test_get_frequency_map_channel(self):
        self.model = create_image_from_visibility(self.vis, npixel=128, cellsize=0.001,
                                                  nchan=self.vnchan,
                                                  frequency=self.startfrequency)
        spectral_mode, vfrequency_map = get_frequency_map(self.vis, self.model)
        assert numpy.max(vfrequency_map) == self.model.nchan - 1
        assert numpy.min(vfrequency_map) == 0
        assert spectral_mode == 'channel'
    
    def test_get_frequency_map_different_channel(self):
        self.model = create_image_from_visibility(self.vis, npixel=128, cellsize=0.001,
                                                  frequency=self.startfrequency, nchan=3,
                                                  channel_bandwidth=2e7)
        spectral_mode, vfrequency_map = get_frequency_map(self.vis, self.model)
        assert numpy.max(vfrequency_map) == self.model.nchan - 1
        assert spectral_mode == 'channel'
    
    def test_get_frequency_map_mfs(self):
        self.model = create_image_from_visibility(self.vis, npixel=128, cellsize=0.001, nchan=1,
                                                  frequency=self.startfrequency)
        spectral_mode, vfrequency_map = get_frequency_map(self.vis, self.model)
        assert numpy.max(vfrequency_map) == 0
        assert spectral_mode == 'mfs'
    
    def test_get_frequency_map_gleam(self):
        self.model = create_low_test_image_from_gleam(npixel=128, cellsize=0.001, frequency=self.frequency,
                                                      channel_bandwidth=self.channel_bandwidth, flux_limit=10.0)
        spectral_mode, vfrequency_map = get_frequency_map(self.vis, self.model)
        assert numpy.max(vfrequency_map) == self.model.nchan - 1
        assert spectral_mode == 'channel'


if __name__ == '__main__':
    unittest.main()
