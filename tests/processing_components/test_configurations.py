"""Unit tests for testing support


"""

import logging
import sys
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from processing_components.simulation.configurations import create_named_configuration
from processing_components.visibility.base import create_visibility

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestConfigurations(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.frequency = numpy.linspace(0.8e8, 1.2e8, 3)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        self.flux = numpy.array([[100.0], [100.0], [100.0]])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
    
    def createVis(self, config, dec=-35.0, rmax=None):
        self.config = create_named_configuration(config, rmax=rmax)
        self.phasecentre = SkyCoord(ra=+15 * u.deg, dec=dec * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_visibility(self.config, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('stokesI'))
    
    def test_named_configurations(self):
        for config in ['LOW', 'LOWBD2', 'LOWBD2-CORE', 'LOWBD1', 'LOWR3', 'ASKAP', 'MID', 'MIDR5']:
            self.createVis(config)
            assert self.config.size() > 0.0
            #print("Config ", config, " has centre", self.config.location.geodetic)
 
        for config in ['LOFAR', 'VLAA', 'VLAA_north']:
            self.createVis(config, +35.0)
            assert self.config.size() > 0.0

    
    def test_SKA_configurations(self):
        for config in ['MID', 'MIDR5', 'LOW', 'LOWR3']:
            self.config = create_named_configuration(config)
            assert self.config.size() > 0.0
    
    def test_clip_configuration(self):
        for rmax in [100.0, 3000.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]:
            self.config = create_named_configuration('LOWBD2', rmax=rmax)
            assert self.config.size() > 0.0
    
    def test_unknown_configuration(self):
        with self.assertRaises(ValueError):
            self.config = create_named_configuration("SKA1-OWL")

if __name__ == '__main__':
    unittest.main()
