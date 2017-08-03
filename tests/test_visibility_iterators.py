"""Unit tests for visibility iterators


"""
import numpy
import unittest

from astropy.coordinates import SkyCoord
import astropy.units as u
from arl.util.testing_support import create_named_configuration
from arl.visibility.iterators import vis_timeslice_iter, vis_wstack_iter, vis_slice_iter
from arl.visibility.base import create_visibility, create_visibility_from_rows

import logging
log = logging.getLogger(__name__)


class TestVisibilityIterators(unittest.TestCase):
    def setUp(self):
        
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        
        self.times = numpy.linspace(-300.0, 300.0, 11) * numpy.pi / 43200.0
        
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

    def test_vis_slice_iterator(self):
        self.actualSetUp()
        for chunk, rows in enumerate(vis_slice_iter(self.vis, step=10000)):
            visslice = create_visibility_from_rows(self.vis, rows)
            assert len(rows)
            assert visslice.vis[0].real == visslice.time[0]

    def test_vis_slice_iterator_vis_slices(self):
        self.actualSetUp()
        for chunk, rows in enumerate(vis_slice_iter(self.vis, vis_slices=11)):
            visslice = create_visibility_from_rows(self.vis, rows)
            assert len(rows)
            assert visslice.vis[0].real == visslice.time[0]

    def test_vis_timeslice_iterator(self):
        self.actualSetUp()
        nchunks = len(list(vis_timeslice_iter(self.vis)))
        log.debug('Found %d chunks' % (nchunks))
        assert nchunks > 1
        for chunk, rows in enumerate(vis_timeslice_iter(self.vis)):
            visslice = create_visibility_from_rows(self.vis, rows)
            assert visslice.vis[0].real == visslice.time[0]
            assert len(rows)
            assert numpy.sum(rows) < self.vis.nvis

    def test_vis_timeslice_iterator_timeslice(self):
        self.actualSetUp()
        for chunk, rows in enumerate(vis_timeslice_iter(self.vis, timeslice=65.0)):
            visslice = create_visibility_from_rows(self.vis, rows)
            assert visslice.vis[0].real == visslice.time[0]
            assert len(rows)
            assert numpy.sum(rows) < self.vis.nvis

    def test_vis_timeslice_iterator_single(self):
        self.actualSetUp(times=numpy.zeros([1]))
        nchunks = len(list(vis_timeslice_iter(self.vis)))
        log.debug('Found %d chunks' % (nchunks))
        for chunk, rows in enumerate(vis_timeslice_iter(self.vis)):
            assert len(rows)

    def test_vis_wstack_iterator(self):
        self.actualSetUp()
        nchunks = len(list(vis_wstack_iter(self.vis, wstack=10.0)))
        log.debug('Found %d chunks' % (nchunks))
        assert nchunks > 1
        for chunk, rows in enumerate(vis_wstack_iter(self.vis, wstack=10.0)):
            assert len(rows)
            visslice = create_visibility_from_rows(self.vis, rows)
            assert numpy.sum(visslice.nvis) < self.vis.nvis

    def test_vis_wstack_iterator_vis_slices(self):
        self.actualSetUp()
        nchunks = len(list(vis_wstack_iter(self.vis, vis_slices=11)))
        assert nchunks == 11
        log.debug('Found %d chunks' % (nchunks))
        assert nchunks > 1
        for chunk, rows in enumerate(vis_wstack_iter(self.vis, vis_slices=11)):
            assert len(rows)
            visslice = create_visibility_from_rows(self.vis, rows)
            assert numpy.sum(visslice.nvis) < self.vis.nvis


if __name__ == '__main__':
    unittest.main()
