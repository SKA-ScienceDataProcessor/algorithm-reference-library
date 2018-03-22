"""Unit tests for visibility selectors


"""
import numpy
import unittest

from astropy.coordinates import SkyCoord
import astropy.units as u
from arl.util.testing_support import create_named_configuration
from arl.visibility.vis_select import vis_select_uvrange
from arl.visibility.base import create_visibility

import logging
log = logging.getLogger(__name__)


class TestVisibilitySelectors(unittest.TestCase):
    def setUp(self):
        
        self.lowcore = create_named_configuration('LOWBD2', rmax=1000.0)
        
        self.times = numpy.linspace(-300.0, 300.0, 11) * numpy.pi / 43200.0
        
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e8])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')

        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                     weight=1.0)

    def test_vis_select_uvrange(self):
        assert self.vis.nvis > numpy.sum(vis_select_uvrange(self.vis, uvmin=50.0, uvmax=60.0))
