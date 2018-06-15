""" Unit tests for skycomponents

"""

import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame

from processing_components.skycomponent.operations import create_skycomponent, find_separation_skycomponents, \
    find_skycomponent_matches, find_nearest_skycomponent, find_nearest_skycomponent_index
from simulation.testing_support import create_low_test_skycomponents_from_gleam

log = logging.getLogger(__name__)


class TestSkycomponent(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e6])
        self.phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
        self.components = create_low_test_skycomponents_from_gleam(flux_limit=2.0,
                                                                   phasecentre=self.phasecentre,
                                                                   frequency=self.frequency,
                                                                   polarisation_frame=PolarisationFrame('stokesI'),
                                                                   radius=0.1)
    
    def test_time_setup(self):
        pass
    
    def test_copy(self):
        fluxes = numpy.linspace(0, 1.0, 10)
        sc = [create_skycomponent(direction=self.phasecentre, flux=numpy.array([[f]]), frequency=self.frequency,
                                  polarisation_frame=PolarisationFrame('stokesI')) for f in fluxes]
        assert len(sc) == len(fluxes)
    
    def test_find_skycomponent_separation(self):
        separations = find_separation_skycomponents(self.components)
        assert separations[0, 0] == 0.0
        assert numpy.max(separations) > 0.0
    
    def test_find_skycomponent_separation_binary(self):
        test = self.components[:len(self.components) // 2]
        separations = find_separation_skycomponents(test, self.components)
        
        assert separations[0, 0] == 0.0
        assert numpy.max(separations) > 0.0
    
    def test_find_skycomponent_matches(self):
        matches = find_skycomponent_matches(self.components[:len(self.components) // 2], self.components)
        assert matches == [(0, 0, 0.0), (1, 1, 0.0), (2, 2, 0.0), (3, 3, 0.0), (4, 4, 0.0), (5, 5, 0.0), (6, 6, 0.0)]
        matches = find_skycomponent_matches(self.components[len(self.components) // 2:], self.components)
        assert matches == [(0, 7, 0.0), (1, 8, 0.0), (2, 9, 0.0), (3, 10, 0.0), (4, 11, 0.0), (5, 12, 0.0),
                           (6, 13, 0.0)]
        matches = find_skycomponent_matches(self.components, self.components[:len(self.components) // 2])
        assert matches == [(0, 0, 0.0), (1, 1, 0.0), (2, 2, 0.0), (3, 3, 0.0), (4, 4, 0.0), (5, 5, 0.0), (6, 6, 0.0)]
        matches = find_skycomponent_matches(self.components, self.components[len(self.components) // 2:])
        assert matches == [(7, 0, 0.0), (8, 1, 0.0), (9, 2, 0.0), (10, 3, 0.0), (11, 4, 0.0), (12, 5, 0.0),
                           (13, 6, 0.0)]
    
    def test_find_nearest_component_index(self):
        match = find_nearest_skycomponent_index(self.components[3].direction, self.components)
        assert match == 3
    
    def test_find_nearest_component(self):
        match, sep = find_nearest_skycomponent(self.components[3].direction, self.components)
        assert match.name == 'GLEAM J021305-474112'


if __name__ == '__main__':
    unittest.main()
