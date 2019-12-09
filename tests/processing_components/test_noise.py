"""Unit tests for testing support


"""

import logging
import sys
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from processing_components.simulation import create_named_configuration
from processing_components.simulation.noise import addnoise_visibility
from processing_components.visibility.base import create_visibility, create_blockvisibility, copy_visibility

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestNoise(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.frequency = numpy.linspace(0.8e8, 1.2e8, 5)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
        self.flux = numpy.array([[100.0], [100.0], [100.0], [100.0], [100.0]])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.config = create_named_configuration('LOWBD2-CORE')
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        nants = self.config.xyz.shape[0]
        assert nants > 1
        assert len(self.config.names) == nants
        assert len(self.config.mount) == nants
    
    def test_addnoise_visibility(self):
        self.vis = create_visibility(self.config, self.times, self.frequency, phasecentre=self.phasecentre,
                                     weight=1.0, polarisation_frame=PolarisationFrame('stokesIQUV'),
                                     channel_bandwidth=self.channel_bandwidth)
        original = copy_visibility(self.vis)
        self.vis = addnoise_visibility(self.vis)
        actual = numpy.std(numpy.abs(self.vis.vis - original.vis))
        assert abs(actual - 0.010786973492702846) < 1e-4, actual
    
    def test_addnoise_blockvisibility(self):
        self.vis = create_blockvisibility(self.config, self.times, self.frequency, phasecentre=self.phasecentre,
                                          weight=1.0, polarisation_frame=PolarisationFrame('stokesIQUV'),
                                          channel_bandwidth=self.channel_bandwidth)
        original = copy_visibility(self.vis)
        self.vis = addnoise_visibility(self.vis)
        actual = numpy.std(numpy.abs(self.vis.vis - original.vis))
        assert abs(actual - 0.01077958403015586) < 1e-4, actual


if __name__ == '__main__':
    unittest.main()
