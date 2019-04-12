"""Unit tests for testing support


"""

import logging
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame

from processing_components.image.operations import export_image_to_fits
from processing_components.imaging.base import create_image_from_visibility
from processing_components.imaging.primary_beams import create_pb, create_vp
from processing_components.simulation.configurations import create_named_configuration
from processing_components.visibility.base import create_visibility

log = logging.getLogger(__name__)


class TestPrimaryBeams(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.frequency = numpy.linspace(1e8, 1.5e8, 3)
        self.channel_bandwidth = numpy.array([2.5e7, 2.5e7, 2.5e7])
        self.flux = numpy.array([[100.0], [100.0], [100.0]])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.config = create_named_configuration('LOWBD2-CORE')
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        nants = self.config.xyz.shape[0]
        assert nants > 1
        assert len(self.config.names) == nants
        assert len(self.config.mount) == nants
    
    def createVis(self, config, dec=-35.0, rmax=None):
        self.config = create_named_configuration(config, rmax=rmax)
        self.phasecentre = SkyCoord(ra=+15 * u.deg, dec=dec * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_visibility(self.config, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('stokesI'))

    def test_create_primary_beams(self):
        self.createVis(config='LOWBD2', rmax=1000.0)
        for telescope in ['VLA', 'ASKAP', 'MID', 'LOW']:
            model = create_image_from_visibility(self.vis, cellsize=0.001, override_cellsize=False)
            beam=create_pb(model, telescope=telescope)
            assert numpy.max(beam.data) > 0.0
            export_image_to_fits(beam, "%s/test_primary_beam_%s.fits" % (self.dir, telescope))
            
    def test_create_voltage_patterns(self):
        self.createVis(config='LOWBD2', rmax=1000.0)
        for telescope in ['VLA', 'ASKAP', 'MID', 'LOW']:
            model = create_image_from_visibility(self.vis, cellsize=0.001, override_cellsize=False)
            beam=create_vp(model, telescope=telescope)
            assert numpy.max(numpy.abs(beam.data.real)) > 0.0
            assert numpy.max(numpy.abs(beam.data.imag)) < 1e-15
