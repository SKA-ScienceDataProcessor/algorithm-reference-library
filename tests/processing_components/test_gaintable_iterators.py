"""Unit tests for visibility iterators


"""
import numpy
import unittest

from astropy.coordinates import SkyCoord
import astropy.units as u

from processing_components.simulation.configurations import create_named_configuration
from processing_components.calibration.iterators import gaintable_timeslice_iter, gaintable_null_iter
from processing_components.calibration.operations import create_gaintable_from_blockvisibility
from processing_components.visibility.base import create_blockvisibility

import logging
log = logging.getLogger(__name__)


class TestGainTableIterators(unittest.TestCase):
    def setUp(self):
        
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        
        self.times = numpy.linspace(-300.0, 300.0, 11) * numpy.pi / 43200.0
        
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e8])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
    
    def actualSetUp(self, times=None):
        if times is not None:
            self.times = times
        
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                     weight=1.0)
        self.gaintable = create_gaintable_from_blockvisibility(self.vis)

    def test_gt_null_iterator(self):
        self.actualSetUp()
        for chunk, rows in enumerate(gaintable_null_iter(self.gaintable)):
            assert chunk<1, "Null iterator returns more than one value"

    def test_gt_timeslice_iterator(self):
        self.actualSetUp()
        nchunks = len(list(gaintable_timeslice_iter(self.gaintable, timeslice='auto')))
        log.debug('Found %d chunks' % (nchunks))
        assert nchunks > 1
        total_rows = 0
        for chunk, rows in enumerate(gaintable_timeslice_iter(self.gaintable, timeslice='auto')):
            total_rows += numpy.sum(rows)
            assert len(rows)
            assert numpy.sum(rows) < self.gaintable.gain.shape[0]
        assert total_rows == self.gaintable.gain.shape[0], \
            "Total rows iterated %d, Original rows %d" % (total_rows, self.gaintable.gain.shape[0])

if __name__ == '__main__':
    unittest.main()
