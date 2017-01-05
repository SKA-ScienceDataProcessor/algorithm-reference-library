"""Unit tests for visibility operations

realtimcornwell@gmail.com
"""

import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.util.testing_support import create_named_configuration
from arl.visibility.iterators import vis_timeslice_iter
from arl.visibility.operations import create_visibility

log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestVisibilityIterators(unittest.TestCase):
    def setUp(self):
        
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        
        times = numpy.arange(-numpy.pi / 4.0, +numpy.pi / 4.0, numpy.pi * 0.5 / 12.0)
        frequency = numpy.array([1e8])
        phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.vis = create_visibility(self.lowcore, times, frequency, weight=1.0, phasecentre=phasecentre)
    
    def test_vis_timeslice_iterator(self):
        params = {}
        params['timeslice'] = 100.0
        timemin = numpy.min(self.vis.time)
        timemax = numpy.max(self.vis.time)
        for rows in vis_timeslice_iter(self.vis, params):
            slicemin = numpy.min(self.vis.data['time'])
            slicemax = numpy.max(self.vis.data['time'])
            assert slicemin >= timemin
            timemin = slicemin
            assert len(rows)
            self.vis.data['vis'][rows,...] = slicemax.astype('complex')
        # Check for writeback
        assert numpy.max(numpy.abs(self.vis.data['vis'])) == timemax, \
            "Write back failed %s %s" % (numpy.max(numpy.abs(self.vis.data['vis'])), timemax)

    def test_vis_timeslice_iterator_all(self):
        params = {}
        params['timeslice'] = 86400.0
        timemin = numpy.min(self.vis.time)
        timemax = numpy.max(self.vis.time)
        for rows in vis_timeslice_iter(self.vis, params):
            slicemin = numpy.min(self.vis.data['time'])
            slicemax = numpy.max(self.vis.data['time'])
            assert slicemin >= timemin
            timemin = slicemin
            assert len(rows)
            self.vis.data['vis'][rows,...] = slicemax.astype('complex')
        # Check for writeback
        assert numpy.max(numpy.abs(self.vis.data['vis'])) == timemax, \
            "Write back failed %s %s" % (numpy.max(numpy.abs(self.vis.data['vis'])), timemax)

if __name__ == '__main__':
    import sys
    import logging
    
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    unittest.main()
