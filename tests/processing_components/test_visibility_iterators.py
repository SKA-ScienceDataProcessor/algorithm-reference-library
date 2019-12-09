"""Unit tests for visibility iterators


"""
import numpy
import unittest

from astropy.coordinates import SkyCoord
import astropy.units as u
from processing_components.simulation import create_named_configuration
from processing_components.visibility.iterators import vis_timeslice_iter, vis_wslice_iter, vis_null_iter, vis_timeslices, vis_wslices
from processing_components.visibility.base import create_visibility, create_visibility_from_rows

import logging
log = logging.getLogger(__name__)


class TestVisibilityIterators(unittest.TestCase):
    def setUp(self):
        
        self.lowcore = create_named_configuration('LOWBD2', rmax=750.0)
        
        self.times = numpy.linspace(-300.0, 300.0, 5) * numpy.pi / 43200.0
        
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e8])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
    
    def actualSetUp(self, times=None):
        if times is not None:
            self.times = times
        
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                     weight=1.0)
        self.vis.data['vis'][:, 0] = self.vis.time

    def test_vis_null_iterator(self):
        self.actualSetUp()
        for chunk, rows in enumerate(vis_null_iter(self.vis)):
            assert chunk<1, "Null iterator returns more than one value"


    def test_vis_timeslice_iterator(self):
        self.actualSetUp()
        nchunks = vis_timeslices(self.vis, timeslice='auto')
        log.debug('Found %d chunks' % (nchunks))
        assert nchunks > 1
        total_rows = 0
        for chunk, rows in enumerate(vis_timeslice_iter(self.vis, nchunks)):
            visslice = create_visibility_from_rows(self.vis, rows)
            total_rows += visslice.nvis
            assert visslice.vis[0].real == visslice.time[0]
            assert len(rows)
            assert numpy.sum(rows) < self.vis.nvis
        assert total_rows == self.vis.nvis, "Total rows iterated %d, Original rows %d" % (total_rows, self.vis.nvis)


    def test_vis_timeslice_iterator_single(self):
        self.actualSetUp(times=numpy.zeros([1]))
        nchunks = vis_timeslices(self.vis, timeslice='auto')
        log.debug('Found %d chunks' % (nchunks))
        for chunk, rows in enumerate(vis_timeslice_iter(self.vis)):
            assert len(rows)

    def test_vis_wslice_iterator(self):
        self.actualSetUp()
        nchunks = vis_wslices(self.vis, wslice=10.0)
        log.debug('Found %d chunks' % (nchunks))
        assert nchunks > 1
        total_rows = 0
        for chunk, rows in enumerate(vis_wslice_iter(self.vis, nchunks)):
            assert len(rows)
            visslice = create_visibility_from_rows(self.vis, rows)
            total_rows += visslice.nvis
            assert numpy.sum(visslice.nvis) < self.vis.nvis
        assert total_rows == self.vis.nvis, "Total rows iterated %d, Original rows %d" % (total_rows, self.vis.nvis)

if __name__ == '__main__':
    unittest.main()
