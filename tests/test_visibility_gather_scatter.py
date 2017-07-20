"""Unit tests for visibility scatter gather


"""

import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.util.testing_support import create_named_configuration

from arl.visibility.gather_scatter import visibility_gather_index, visibility_gather_time, visibility_gather_w, \
    visibility_scatter_index, visibility_scatter_time, visibility_scatter_w, visibility_scatter_channel, \
    visibility_gather_channel
from arl.visibility.base import create_visibility, create_blockvisibility

import logging

log = logging.getLogger(__name__)


class TestVisibilityGatherScatter(unittest.TestCase):
    
    def setUp(self):
    
        self.lowcore = create_named_configuration('LOWBD2-CORE')
    
        self.times = numpy.linspace(-300.0, 300.0, 11) * numpy.pi / 43200.0
    
        self.frequency = numpy.linspace(1e8, 1.5e9, 7)
        
        self.channel_bandwidth = numpy.array(7*[self.frequency[1]-self.frequency[0]])
        
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')

    def actualSetUp(self, times=None):
        if times is not None:
            self.times = times
            
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                     weight=1.0)
        self.vis.data['vis'][:,0] = self.vis.time
        self.blockvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                     weight=1.0)
        self.blockvis.data['vis'][...] = 1.0

    def test_vis_scatter_gather_slice(self):
        self.actualSetUp()
        vis_list = visibility_scatter_index(self.vis, step=1000)
        newvis = visibility_gather_index(vis_list, self.vis, step=1000)
        assert self.vis.nvis == newvis.nvis
        assert numpy.max(numpy.abs(newvis.vis)) > 0.0


    def test_vis_scatter_gather_wstack(self):
        self.actualSetUp()
        vis_list = visibility_scatter_w(self.vis, wstack=10.0)
        newvis = visibility_gather_w(vis_list, self.vis, wstack=10.0)
        assert self.vis.nvis == newvis.nvis
        assert numpy.max(numpy.abs(newvis.vis)) > 0.0


    def test_vis_scatter_gather_vis_slices(self):
        self.actualSetUp()
        vis_list = visibility_scatter_w(self.vis, vis_slices=11)
        newvis = visibility_gather_w(vis_list, self.vis, vis_slices=11)
        assert self.vis.nvis == newvis.nvis
        assert numpy.max(numpy.abs(newvis.vis)) > 0.0

    def test_vis_scatter_gather_timeslice(self):
        self.actualSetUp()
        vis_list = visibility_scatter_time(self.vis)
        newvis = visibility_gather_time(vis_list, self.vis)
        assert self.vis.nvis == newvis.nvis
        assert numpy.max(numpy.abs(newvis.vis)) > 0.0

    def test_vis_scatter_gather_channel(self):
        self.actualSetUp()
        vis_list = visibility_scatter_channel(self.blockvis)
        assert len(vis_list) == len(self.blockvis.frequency)
        assert vis_list[0].vis.shape[-2] == 1
        assert numpy.max(numpy.abs(vis_list[0].vis)) > 0.0
        newvis = visibility_gather_channel(vis_list, self.blockvis)
        assert len(newvis.frequency) == len(self.blockvis.frequency)
        assert self.blockvis.nvis == newvis.nvis
        assert numpy.max(numpy.abs(newvis.vis)) > 0.0

    def test_vis_scatter_gather_channel_None(self):
        self.actualSetUp()
        vis_list = visibility_scatter_channel(self.blockvis)
        assert len(vis_list) == len(self.blockvis.frequency)
        assert vis_list[0].vis.shape[-2] == 1
        assert numpy.max(numpy.abs(vis_list[0].vis)) > 0.0
        newvis = visibility_gather_channel(vis_list)
        assert len(newvis.frequency) == len(self.blockvis.frequency)
        assert self.blockvis.nvis == newvis.nvis
        assert numpy.max(numpy.abs(newvis.vis)) > 0.0


if __name__ == '__main__':
    unittest.main()
