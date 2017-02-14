"""Unit tests for visibility operations

realtimcornwell@gmail.com
"""

import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.util.testing_support import create_named_configuration, run_unittests
from arl.visibility.iterators import vis_timeslice_iter
from arl.visibility.operations import create_compressedvisibility, create_compressedvisibility_from_rows

import logging

log = logging.getLogger(__name__)


class TestVisibilityIterators(unittest.TestCase):
    def setUp(self):
        
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        
        times = numpy.linspace(-numpy.pi / 4.0, +numpy.pi / 4.0, 12)
        frequency = numpy.array([1e8])
        phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.vis = create_compressedvisibility(self.lowcore, times, frequency, phasecentre=phasecentre, weight=1.0)
        self.vis.data['vis'] = self.vis.time
    
    def test_vis_timeslice_iterator(self):
        nchunks = len(list(vis_timeslice_iter(self.vis)))
        log.debug('Found %d chunks' % (nchunks))
        assert nchunks > 1
        for chunk, rows in enumerate(vis_timeslice_iter(self.vis)):
            visslice = create_compressedvisibility_from_rows(self.vis, rows)
            assert visslice.vis[0].real == visslice.time[0]
            assert len(rows)
            assert numpy.sum(rows) < self.vis.nvis
            
if __name__ == '__main__':
    run_unittests()
