"""Unit tests for testing support


"""

import logging
import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.data.polarisation import PolarisationFrame
from arl.image.operations import export_image_to_fits
from arl.imaging.base import create_image_from_visibility
from arl.util.primary_beams import create_pb_vla
from arl.util.testing_support import create_named_configuration
from arl.visibility.base import create_visibility

log = logging.getLogger(__name__)


class TestPrimaryBeams(unittest.TestCase):
    def setUp(self):
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        
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

    def test_create_primary_beams_vla(self):
        self.createVis(config='LOWBD2', rmax=1000.0)
        model = create_image_from_visibility(self.vis, cellsize=0.00001, override_cellsize=True)
        beam=create_pb_vla(model)
        assert numpy.max(beam.data) > 0.0
        export_image_to_fits(beam, "%s/primary_beam_vla.fits" % self.dir)