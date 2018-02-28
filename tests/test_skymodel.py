""" Unit tests for skymodel

"""

import logging
import os
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from arl.calibration.operations import create_gaintable_from_blockvisibility
from arl.data.polarisation import PolarisationFrame
from arl.data.skymodel import SkyModel
from arl.skycomponent.operations import create_skycomponent
from arl.util.testing_support import create_test_image, create_named_configuration
from arl.visibility.base import create_blockvisibility

log = logging.getLogger(__name__)


class TestSkyModel(unittest.TestCase):
    def setUp(self):
        self.dir = './test_results'
        self.lowcore = create_named_configuration('LOWBD2', rmax=300.0)
        os.makedirs(self.dir, exist_ok=True)
        self.times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 7)
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e6])
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre, weight=1.0,
                                          polarisation_frame=PolarisationFrame('stokesI'))
        self.vis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_test_image(cellsize=0.0015, phasecentre=self.vis.phasecentre,
                                       frequency=self.frequency)
        self.model.data[self.model.data > 1.0] = 1.0
    
    def test_create(self):
        gt = create_gaintable_from_blockvisibility(self.vis)
        fluxes = numpy.linspace(0, 1.0, 11)
        sc = [create_skycomponent(direction=self.phasecentre, flux=numpy.array([[f]]),
                                  frequency=self.frequency,
                                  polarisation_frame=PolarisationFrame('stokesI')) for f in fluxes]
        sm = SkyModel(images=[self.model], components=sc, gt=gt)
        print(sm)
