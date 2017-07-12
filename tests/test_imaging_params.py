"""Unit tests for Fourier transform processor params


"""
import logging
import os
import unittest

import numpy
from arl.imaging.params import get_frequency_map
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.data.polarisation import PolarisationFrame
from arl.util.testing_support import create_named_configuration, create_low_test_image_from_s3, \
    create_low_test_image_from_gleam
from arl.visibility.operations import create_visibility
from arl.imaging import create_image_from_visibility

log = logging.getLogger(__name__)


class TestFTProcessorParams(unittest.TestCase):
    def setUp(self):
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)

        self.vnchan = 5
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 7)
        self.frequency = numpy.linspace(8e7, 1.2e8, self.vnchan)
        self.startfrequency = numpy.array([8e7])
        self.channel_bandwidth = numpy.array(self.vnchan * [self.frequency[1]-self.frequency[0]])
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_visibility(self.lowcore, times=self.times, frequency=self.frequency,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('stokesI'),
                                     channel_bandwidth=self.channel_bandwidth)
        self.model = create_image_from_visibility(self.vis, npixel=256, cellsize=0.001, nchan=self.vnchan,
                                                  frequency=self.startfrequency)
    def test_get_frequency_map_channel(self):
        self.model = create_image_from_visibility(self.vis, npixel=256, cellsize=0.001, nchan=self.vnchan,
                                                  frequency=self.startfrequency)
        spectral_mode, vfrequency_map = get_frequency_map(self.vis, self.model)
        assert numpy.max(vfrequency_map) == self.model.nchan - 1
        assert spectral_mode == 'channel'

    def test_get_frequency_map_different_channel(self):
        self.model = create_image_from_visibility(self.vis, npixel=256, cellsize=0.001,
                                                  frequency=self.startfrequency, nchan=3,
                                                  channel_bandwidth=2e7)
        spectral_mode, vfrequency_map = get_frequency_map(self.vis, self.model)
        assert numpy.max(vfrequency_map) == self.model.nchan - 1
        assert spectral_mode == 'channel'

    def test_get_frequency_map_mfs(self):
        self.model = create_image_from_visibility(self.vis, npixel=256, cellsize=0.001, nchan=1,
                                                  frequency=self.startfrequency)
        spectral_mode, vfrequency_map = get_frequency_map(self.vis, self.model)
        assert numpy.max(vfrequency_map) == 0
        assert spectral_mode == 'mfs'

    def test_get_frequency_map_gleam(self):
        self.model = create_low_test_image_from_gleam(npixel=256, cellsize=0.001, frequency=self.frequency,
                                                      channel_bandwidth=self.channel_bandwidth)
        spectral_mode, vfrequency_map = get_frequency_map(self.vis, self.model)
        assert numpy.max(vfrequency_map) == self.model.nchan - 1
        assert spectral_mode == 'channel'


    def test_get_frequency_map_s3(self):
        self.model = create_low_test_image_from_s3(npixel=64, cellsize=0.001, frequency=self.frequency,
                                                   channel_bandwidth=self.channel_bandwidth)
        spectral_mode, vfrequency_map = get_frequency_map(self.vis, self.model)
        assert numpy.max(vfrequency_map) == self.model.nchan - 1
        assert spectral_mode == 'channel'


if __name__ == '__main__':
    unittest.main()
