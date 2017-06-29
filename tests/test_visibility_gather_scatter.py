"""Unit tests for visibility scatter gather


"""

import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.util.testing_support import create_named_configuration

from arl.visibility.gather_scatter import visibility_gather_index, visibility_gather_time, visibility_gather_w, \
    visibility_scatter_index, visibility_scatter_time, visibility_scatter_w
from arl.visibility.operations import create_visibility, create_visibility_from_rows

import logging

log = logging.getLogger(__name__)


class TestVisibilityGatherScatter(unittest.TestCase):
    
    def setUp(self):
    
        self.lowcore = create_named_configuration('LOWBD2-CORE')
    
        self.times = numpy.linspace(-300.0, 300.0, 11) * numpy.pi / 43200.0
    
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e8])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)

    def actualSetUp(self, times=None):
        if times is not None:
            self.times = times
            
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                     weight=1.0)
        self.vis.data['vis'][:,0] = self.vis.time

    def test_vis_scatter_gather_slice(self):
        self.actualSetUp()
        vis_list = visibility_scatter_index(self.vis, step=1000)
        newvis = visibility_gather_index(vis_list, self.vis, step=1000)
        assert self.vis.nvis == newvis.nvis


    def test_vis_scatter_gather_wstack(self):
        self.actualSetUp()
        vis_list = visibility_scatter_w(self.vis, wstack=10.0)
        newvis = visibility_gather_w(vis_list, self.vis, wstack=10.0)
        assert self.vis.nvis == newvis.nvis

    def test_vis_scatter_gather_timeslice(self):
        self.actualSetUp()
        vis_list = visibility_scatter_time(self.vis)
        newvis = visibility_gather_time(vis_list, self.vis)
        assert self.vis.nvis == newvis.nvis


if __name__ == '__main__':
    unittest.main()
