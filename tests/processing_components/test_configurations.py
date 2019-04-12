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
        
        self.frequency = numpy.linspace(0.8e8, 1.2e8, 5)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
        self.flux = numpy.array([[100.0], [100.0], [100.0], [100.0], [100.0]])
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
        for config in ['LOWBD2', 'LOWBD2-CORE', 'LOWBD1', 'LOFAR', 'ASKAP', 'LOWR3', 'MIDR5']:
            self.createVis(config)
            assert self.config.size() > 0.0
            print("Config ", config, " has centre", self.config.location.geodetic)
        
        self.createVis('VLAA', +35.0)
        self.createVis('VLAA_north', +35.0)
    
    def test_SKA_configurations(self):
        for config in ['MIDR5', 'LOWR3']:
            self.config = create_named_configuration(config)
            assert self.config.size() > 0.0
    
    def test_clip_configuration(self):
        for rmax in [100.0, 3000.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]:
            self.config = create_named_configuration('LOWBD2', rmax=rmax)
            assert self.config.size() > 0.0
    
    def test_unknown_configuration(self):
        with self.assertRaises(ValueError):
            self.config = create_named_configuration("SKA1-OWL")

    def test_plot_LOWs(self):
        lowbd2 = create_named_configuration('LOWBD2')
        askap = create_named_configuration('ASKAP')
        lowr3 = create_named_configuration('LOWR3')
    
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(lowbd2.xyz[:, 0], lowbd2.xyz[:, 1], '.')
        plt.plot(lowr3.xyz[:, 0], lowr3.xyz[:, 1], '.')
        plt.plot(askap.xyz[:, 0], askap.xyz[:, 1], '.')
        plt.show()

    def tesplot_MID(self):
        mid = create_named_configuration('MID')
        midr5 = create_named_configuration('MIDR5')

        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(mid.xyz[:, 0], mid.xyz[:, 1], '.')
        plt.plot(midr5.xyz[:, 0], midr5.xyz[:, 1], '.')
        plt.show()


if __name__ == '__main__':
    unittest.main()
