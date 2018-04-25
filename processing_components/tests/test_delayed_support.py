""" Unit tests for testing support


"""
import logging
import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from libs.data.data_models import BlockVisibility
from processing_components.graphs.graph_support import create_simulate_vis_graph
from processing_components.graphs.execute import arlexecute

log = logging.getLogger(__name__)


class TestTestingDaskGraphSupport(unittest.TestCase):
    def setUp(self):
    
        from libs.data.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.frequency = numpy.linspace(1e8, 1.5e8, 3)
        self.channel_bandwidth = numpy.array([2.5e7, 2.5e7, 2.5e7])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
    
    def test_create_simulate_vis_graph(self):
        arlexecute.set_client(use_dask=True)
        vis_graph_list = create_simulate_vis_graph(frequency=self.frequency, channel_bandwidth=self.channel_bandwidth)
        assert len(vis_graph_list) == len(self.frequency)
        vt = vis_graph_list[0].compute()
        assert isinstance(vt, BlockVisibility)
        assert vt.nvis > 0
        arlexecute.client.close()
 
