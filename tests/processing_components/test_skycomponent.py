""" Unit tests for skycomponents

"""

import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from processing_library.image.operations import create_image
from processing_components.simulation.testing_support import create_low_test_skycomponents_from_gleam
from processing_components.skycomponent.operations import create_skycomponent, find_separation_skycomponents, \
    find_skycomponent_matches, find_nearest_skycomponent, find_nearest_skycomponent_index, \
    filter_skycomponents_by_flux, select_neighbouring_components, voronoi_decomposition

log = logging.getLogger(__name__)


class TestSkycomponent(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e6])
        self.phasecentre = SkyCoord(ra=+0.0 * u.deg, dec=-55.0 * u.deg, frame='icrs', equinox='J2000')
        self.components = create_low_test_skycomponents_from_gleam(flux_limit=0.1,
                                                                   phasecentre=self.phasecentre,
                                                                   frequency=self.frequency,
                                                                   polarisation_frame=PolarisationFrame('stokesI'),
                                                                   radius=0.5)

    def test_time_setup(self):
        pass
    
    def test_filter_flux(self):
        newsc = filter_skycomponents_by_flux(self.components, flux_min=0.3)
        assert len(newsc) < len(self.components), len(self.components)
        newsc = filter_skycomponents_by_flux(self.components, flux_min=5.0)
        assert len(newsc) == 138, len(newsc)
        newsc = filter_skycomponents_by_flux(self.components, flux_max=8.0)
        assert len(newsc) == 11744, len(newsc)

    def test_copy(self):
        fluxes = numpy.linspace(0, 1.0, 10)
        sc = [create_skycomponent(direction=self.phasecentre, flux=numpy.array([[f]]), frequency=self.frequency,
                                  polarisation_frame=PolarisationFrame('stokesI')) for f in fluxes]
        assert len(sc) == len(fluxes)
    
    def test_find_skycomponent_separation(self):
        separations = find_separation_skycomponents(self.components[0:99])
        assert separations[0, 0] == 0.0
        assert numpy.max(separations) > 0.0
    
    def test_find_skycomponent_separation_binary(self):
        test = self.components[0:9]
        separations = find_separation_skycomponents(test, test)
        
        assert separations[0, 0] == 0.0
        assert numpy.max(separations) > 0.0
    
    def test_find_skycomponent_matches(self):
        matches = find_skycomponent_matches(self.components[:len(self.components) // 2], self.components)
        assert len(matches) == 5900, len(matches)
    
    def test_find_nearest_component_index(self):
        match = find_nearest_skycomponent_index(self.components[3].direction, self.components)
        assert match == 3
    
    def test_find_nearest_component(self):
        match, sep = find_nearest_skycomponent(self.components[3].direction, self.components)
        assert match.name == 'GLEAM J211146-685527', match.name
        
    def test_select_neighbouring_components(self):
        bright_components = filter_skycomponents_by_flux(self.components, flux_min=2.0)
        indices, d2d = select_neighbouring_components(self.components, bright_components)
        assert len(indices) == 11801, len(indices)
        assert numpy.max(indices) == (len(bright_components) - 1)

    def test_voronoi_decomposition(self):
        bright_components = filter_skycomponents_by_flux(self.components, flux_min=2.0)
        model = create_image(npixel=512, phasecentre=self.phasecentre, frequency=self.frequency,
                             polarisation_frame=PolarisationFrame('stokesI'))
        vor, vor_image = voronoi_decomposition(model, bright_components)
        print(vor_image)


if __name__ == '__main__':
    unittest.main()
