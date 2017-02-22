"""Unit tests for visibility operations

realtimcornwell@gmail.com
"""

import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.util.testing_support import create_named_configuration
from arl.util.run_unittests import run_unittests

from arl.visibility.iterators import *
from arl.visibility.operations import create_visibility, create_visibility_from_rows

import logging

log = logging.getLogger(__name__)


class TestVisibilityIterators(unittest.TestCase):
    
    def setUp(self):
    
        self.lowcore = create_named_configuration('LOWBD2-CORE')
    
        self.times = numpy.linspace(-300.0, 300.0, 11) * numpy.pi / 43200.0
    
        self.frequency = numpy.array([1e8])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)

    def actualSetUp(self, times=None):
        if times is not None:
            self.times = times
            
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre, weight=1.0)
        self.vis.data['vis'] = self.vis.time

    def test_vis_slice_iterator(self):
        self.actualSetUp()
        for chunk, rows in enumerate(vis_slice_iter(self.vis, step=10000)):
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

    def test_vis_timeslice_iterator_single(self):
        self.actualSetUp(times=numpy.zeros([1]))
        nchunks = len(list(vis_timeslice_iter(self.vis)))
        log.debug('Found %d chunks' % (nchunks))
        for chunk, rows in enumerate(vis_timeslice_iter(self.vis)):
            visslice = create_visibility_from_rows(self.vis, rows)
            assert visslice.vis[0].real == visslice.time[0]
            assert len(rows)

    def test_vis_wslice_iterator(self):
        self.actualSetUp()
        nchunks = len(list(vis_wslice_iter(self.vis, wslice=1.0)))
        log.debug('Found %d chunks' % (nchunks))
        assert nchunks > 1
        for chunk, rows in enumerate(vis_wslice_iter(self.vis, wslice=1.0)):
            assert len(rows)
            visslice = create_visibility_from_rows(self.vis, rows)
            assert numpy.sum(visslice.nvis) < self.vis.nvis
    


if __name__ == '__main__':
    run_unittests()
