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
        
    def createVis(self, config='MID', dec=-35.0, rmax=1e3, freq=1e9):
        self.frequency = numpy.linspace(freq, 1.5*freq, 3)
        self.channel_bandwidth = numpy.array([2.5e7, 2.5e7, 2.5e7])
        self.flux = numpy.array([[100.0], [100.0], [100.0]])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.config = create_named_configuration(config)
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        nants = self.config.xyz.shape[0]
        assert nants > 1
        assert len(self.config.names) == nants
        assert len(self.config.mount) == nants
    
        self.config = create_named_configuration(config, rmax=rmax)
        self.phasecentre = SkyCoord(ra=+15 * u.deg, dec=dec * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_visibility(self.config, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('stokesI'))

    def test_create_primary_beams(self):
        self.createVis()
        for telescope in ['VLA', 'ASKAP', 'MID', 'LOW']:
            model = create_image_from_visibility(self.vis, cellsize=0.001, override_cellsize=False)
            beam=create_pb(model, telescope=telescope)
            assert numpy.max(beam.data) > 0.0
            export_image_to_fits(beam, "%s/test_primary_beam_%s.fits" % (self.dir, telescope))
            
    def test_create_voltage_patterns(self):
        self.createVis()
        for telescope in ['VLA', 'ASKAP', 'LOW']:
            model = create_image_from_visibility(self.vis, cellsize=0.001, override_cellsize=False)
            beam=create_vp(model, telescope=telescope)
            assert numpy.max(numpy.abs(beam.data.real)) > 0.0
            assert numpy.max(numpy.abs(beam.data.imag)) < 1e-15, numpy.max(numpy.abs(beam.data.imag))

    def test_create_voltage_patterns_numeric(self):
        self.createVis(freq=1.4e9)
        model = create_image_from_visibility(self.vis, npixel=512, cellsize=0.0004, override_cellsize=False)
        pointingcentre = SkyCoord(ra=+17.0 * u.deg, dec=-37.0 * u.deg, frame='icrs', equinox='J2000')
        for telescope in ['MID']:
            beam=create_vp(model, telescope=telescope, padding=4, pointingcentre=pointingcentre)
            beam_data = beam.data
            beam.data = numpy.real(beam_data)
            export_image_to_fits(beam, "%s/test_primary_beam_numeric_real_%s.fits" % (self.dir, telescope))
            beam.data = numpy.imag(beam_data)
            export_image_to_fits(beam, "%s/test_primary_beam_numeric_imag_%s.fits" % (self.dir, telescope))

            beam=create_vp(model, telescope=telescope, numeric=False, pointingcentre=pointingcentre)
            beam.data = numpy.real(beam.data)
            export_image_to_fits(beam, "%s/test_primary_beam_analytic_%s.fits" % (self.dir, telescope))
