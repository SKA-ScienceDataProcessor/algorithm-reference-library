"""Unit tests for visibility operations

realtimcornwell@gmail.com
"""

import unittest

from astropy import units as u
from numpy.testing import assert_allclose

import sys

import numpy

from astropy.coordinates import SkyCoord
from astropy import units as u

from arl.visibility.operations import create_visibility
from arl.visibility.iterators import vis_timeslice_iter
from arl.util.testing_support import create_named_configuration



import logging

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
        for visslice in vis_timeslice_iter(self.vis, params):
            slicemin = numpy.min(visslice.time)
            assert slicemin >= timemin
            timemin = slicemin
            assert len(visslice.data)

    def test_vis_slice_iterator(self):
        params = {}
        params['vissliceaxis'] = 'time'
        params['visslice'] = 100.0

        for visslice in vis_timeslice_iter(self.vis, params):
            assert len(visslice.data)


if __name__ == '__main__':
    import sys
    import logging
    
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    unittest.main()
