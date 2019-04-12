""" Unit tests for mpc

"""

import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.memory_data_models import SkyModel, Image
from data_models.polarisation import PolarisationFrame
from processing_components.imaging.primary_beams import create_low_test_beam
from processing_components.skymodel.operations import expand_skymodel_by_skycomponents
from processing_components.simulation.testing_support import create_low_test_skycomponents_from_gleam
from processing_components.simulation.configurations import create_named_configuration
from processing_components.simulation.testing_support import create_test_image
from processing_components.skycomponent.operations import apply_beam_to_skycomponent, remove_neighbouring_components
from processing_components.skycomponent.operations import filter_skycomponents_by_flux
from processing_components.skymodel.operations import image_voronoi_iter
from processing_components.visibility.base import create_blockvisibility
from processing_library.image.operations import create_image

log = logging.getLogger(__name__)


class TestSkymodelMPC(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        dec = -40.0 * u.deg
        
        self.lowcore = create_named_configuration('LOWBD2', rmax=300.0)
        self.dir = arl_path('test_results')
        self.times = numpy.linspace(-10.0, 10.0, 3) * numpy.pi / (3600.0 * 12.0)
        
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e6])
        self.phasecentre = SkyCoord(ra=+0.0 * u.deg, dec=dec, frame='icrs', equinox='J2000')
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre, weight=1.0,
                                          polarisation_frame=PolarisationFrame('stokesI'))
        self.vis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_image(npixel=512, cellsize=0.000015, polarisation_frame=PolarisationFrame("stokesI"),
                                  frequency=self.frequency, channel_bandwidth=self.channel_bandwidth,
                                  phasecentre=self.phasecentre)
    
    def test_expand_skymodel_by_skycomponents(self):
        beam = create_test_image(cellsize=0.0015, phasecentre=self.vis.phasecentre,
                                 frequency=self.frequency)
        
        beam = create_low_test_beam(beam)
        
        gleam_components = create_low_test_skycomponents_from_gleam(flux_limit=1.0,
                                                                    phasecentre=self.phasecentre,
                                                                    frequency=self.frequency,
                                                                    polarisation_frame=PolarisationFrame(
                                                                        'stokesI'),
                                                                    radius=0.2)
        
        pb_gleam_components = apply_beam_to_skycomponent(gleam_components, beam)
        
        actual_components = filter_skycomponents_by_flux(pb_gleam_components, flux_min=1.0)
        
        assert len(actual_components) == 37, len(actual_components)
        sm = SkyModel(image=self.model, components=actual_components)
        assert len(sm.components) == len(actual_components)
        
        scatter_sm = expand_skymodel_by_skycomponents(sm)
        assert len(scatter_sm) == len(actual_components) + 1
        assert len(scatter_sm[0].components) == 1
    
    def test_expand_skymodel_voronoi(self):
        self.model = create_image(npixel=256, cellsize=0.001, polarisation_frame=PolarisationFrame("stokesI"),
                                  frequency=self.frequency, channel_bandwidth=self.channel_bandwidth,
                                  phasecentre=self.phasecentre)
        beam = create_low_test_beam(self.model)
        
        gleam_components = create_low_test_skycomponents_from_gleam(flux_limit=1.0,
                                                                    phasecentre=self.phasecentre,
                                                                    frequency=self.frequency,
                                                                    polarisation_frame=PolarisationFrame(
                                                                        'stokesI'),
                                                                    radius=0.1)
        
        pb_gleam_components = apply_beam_to_skycomponent(gleam_components, beam)
        
        actual_components = filter_skycomponents_by_flux(pb_gleam_components, flux_min=1.0)
        _, actual_components = remove_neighbouring_components(actual_components, 0.05)
        
        for imask, mask in enumerate(image_voronoi_iter(self.model, actual_components)):
            mask.data *= beam.data
            assert isinstance(mask, Image)
            assert mask.data.dtype == "float"
            assert numpy.sum(mask.data) > 1
            # import matplotlib.pyplot as plt
            # from processing_components.image.operations import show_image
            # show_image(mask)
            # plt.show()

        assert len(actual_components) == 9, len(actual_components)
        sm = SkyModel(image=self.model, components=actual_components)
        assert len(sm.components) == len(actual_components)
        
        scatter_sm = expand_skymodel_by_skycomponents(sm)
        assert len(scatter_sm) == len(actual_components) + 1
        assert len(scatter_sm[0].components) == 1
