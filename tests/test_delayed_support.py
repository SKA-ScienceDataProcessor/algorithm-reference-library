""" Unit tests for testing support


"""
import logging
import os
import unittest

from arl.graphs.execute import arlexecute

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.data.data_models import BlockVisibility
from arl.util.delayed_support import create_simulate_vis_graph

log = logging.getLogger(__name__)


class TestTestingDaskGraphSupport(unittest.TestCase):
    def setUp(self):
    
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        
        self.frequency = numpy.linspace(1e8, 1.5e8, 3)
        self.channel_bandwidth = numpy.array([2.5e7, 2.5e7, 2.5e7])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
    
    def test_create_simulate_vis_graph(self):
        for arlexecute.use_dask in [True, False]:
            vis_graph_list = create_simulate_vis_graph(frequency=self.frequency, channel_bandwidth=self.channel_bandwidth)
            assert len(vis_graph_list) == len(self.frequency)
            vt = arlexecute.get(vis_graph_list[0])
            assert isinstance(vt, BlockVisibility)
            assert vt.nvis > 0
