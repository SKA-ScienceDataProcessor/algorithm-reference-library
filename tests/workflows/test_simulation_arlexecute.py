""" Unit tests for testing support


"""
import logging
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from tests.workflows import ARLExecuteTestCase
from data_models.memory_data_models import BlockVisibility
from wrappers.arlexecute.execution_support.arlexecute import arlexecute

from workflows.arlexecute.simulation.simulation_arlexecute import simulate_list_arlexecute_workflow

log = logging.getLogger(__name__)


class TestSimulationArlexecuteSupport(ARLExecuteTestCase, unittest.TestCase):
    def setUp(self):
        super(TestSimulationArlexecuteSupport, self).setUp()
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.frequency = numpy.linspace(1e8, 1.5e8, 3)
        self.channel_bandwidth = numpy.array([2.5e7, 2.5e7, 2.5e7])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0

    def test_create_simulate_vis_list(self):
        vis_list = simulate_list_arlexecute_workflow(frequency=self.frequency, channel_bandwidth=self.channel_bandwidth)
        assert len(vis_list) == len(self.frequency)
        vt = arlexecute.compute(vis_list[0], sync=True)
        assert isinstance(vt, BlockVisibility)
        assert vt.nvis > 0
 
