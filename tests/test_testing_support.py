"""Unit tests for visibility operations

realtimcornwell@gmail.com
"""

import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.util.testing_support import *
from arl.visibility.iterators import *
from arl.visibility.operations import create_visibility, create_visibility_from_rows

import logging

log = logging.getLogger(__name__)


class TestTesting_Support(unittest.TestCase):
    def setUp(self):
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
    
    def createVis(self, config, dec = -35.0):
        
        self.config = create_named_configuration(config)
        nants = self.config.xyz.shape[0]
        assert nants > 1
        assert len(self.config.names) == nants
        assert len(self.config.mount) == nants


        times = numpy.linspace(-300.0, 300.0, 11) * numpy.pi / 43200.0
        frequency = numpy.array([1e8])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=dec * u.deg, frame='icrs', equinox=2000.0)
        self.vis = create_visibility(self.config, times, frequency, phasecentre=self.phasecentre, weight=1.0)
        
    def test_named_configurations(self):
        for config in ['LOWBD2', 'LOWBD2-CORE', 'LOWBD1', 'LOFAR']:
            self.createVis(config)
        
        self.createVis('VLAA', +35.0)
        self.createVis('VLAA_north', +35.0)

    def test_create_test_image(self):
        im = create_test_image(canonical=False)
        assert len(im.data.shape) == 2
        im = create_test_image(canonical=True)
        assert len(im.data.shape) == 4
        im = create_test_image(canonical=True, npol=4, nchan=3)
        assert len(im.data.shape) == 4
        assert im.data.shape[0] == 3
        assert im.data.shape[1] == 4

    def test_create_low_test_image(self):
        im = create_low_test_image(npixel=1024, channelwidth=1e5, nchan=1,
                                   phasecentre=self.phasecentre)
        assert im.data.shape[0] == 1
        assert im.data.shape[1] == 1
        assert im.data.shape[2] == 1024
        assert im.data.shape[3] == 1024

if __name__ == '__main__':
    run_unittests()
