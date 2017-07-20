"""Unit tests for Visibility coalescence


"""
import unittest

import numpy

from astropy.coordinates import SkyCoord
import astropy.units as u
from arl.data.polarisation import PolarisationFrame
from arl.util.testing_support import create_named_configuration
from arl.visibility.coalesce import coalesce_visibility, decoalesce_visibility, \
    convert_blockvisibility_to_visibility
from arl.visibility.base import create_blockvisibility, create_visibility_from_rows
from arl.visibility.iterators import vis_timeslice_iter

import logging

log = logging.getLogger(__name__)


class TestCoalesce(unittest.TestCase):
    def setUp(self):

        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 30 * 3.76, 3.76)
        df = 27343.75000
        self.frequency = numpy.array([1e8-df, 1e8, 1e8+df])
        self.channel_bandwidth = numpy.array([27343.75, 27343.75, 27343.75])
        self.phasecentre = SkyCoord(ra=+0.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.blockvis = create_blockvisibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                               weight=1.0, polarisation_frame=PolarisationFrame('stokesI'),
                                               channel_bandwidth=self.channel_bandwidth)

    def test_coalesce_decoalesce_zero(self):
        cvis = coalesce_visibility(self.blockvis, time_coal=0.0, frequency_coal=0.0)
        assert numpy.min(cvis.frequency) == numpy.min(self.frequency)
        assert numpy.min(cvis.frequency) > 0.0
        dvis = decoalesce_visibility(cvis)
        assert dvis.nvis == self.blockvis.nvis
        dvis = decoalesce_visibility(cvis, overwrite=True)
        assert dvis.nvis == self.blockvis.nvis

    def test_convert_decoalesce_zero(self):
        cvis = convert_blockvisibility_to_visibility(self.blockvis)
        assert numpy.min(cvis.frequency) == numpy.min(self.frequency)
        assert numpy.min(cvis.frequency) > 0.0
        dvis = decoalesce_visibility(cvis)
        assert dvis.nvis == self.blockvis.nvis
        dvis = decoalesce_visibility(cvis, overwrite=True)
        assert dvis.nvis == self.blockvis.nvis


    def test_coalesce_decoalesce(self):
        cvis = coalesce_visibility(self.blockvis, time_coal=1.0, frequency_coal=1.0)
        assert numpy.min(cvis.frequency) == numpy.min(self.frequency)
        assert numpy.min(cvis.frequency) > 0.0
        dvis = decoalesce_visibility(cvis)
        assert dvis.nvis == self.blockvis.nvis
        dvis = decoalesce_visibility(cvis, overwrite=True)
        assert dvis.nvis == self.blockvis.nvis

    def test_coalesce_decoalesce_frequency(self):
        cvis = coalesce_visibility(self.blockvis, time_coal=0.0, max_time_coal=1, frequency_coal=1.0)
        assert numpy.min(cvis.frequency) == numpy.min(self.frequency)
        assert numpy.min(cvis.frequency) > 0.0
        dvis = decoalesce_visibility(cvis)
        assert dvis.nvis == self.blockvis.nvis
        dvis = decoalesce_visibility(cvis, overwrite=True)
        assert dvis.nvis == self.blockvis.nvis

    def test_coalesce_decoalesce_time(self):
        cvis = coalesce_visibility(self.blockvis, time_coal=1.0, frequency_coal=0.0, max_frequency_coal=1)
        assert numpy.min(cvis.frequency) == numpy.min(self.frequency)
        assert numpy.min(cvis.frequency) > 0.0
        dvis = decoalesce_visibility(cvis)
        assert dvis.nvis == self.blockvis.nvis
        dvis = decoalesce_visibility(cvis, overwrite=True)
        assert dvis.nvis == self.blockvis.nvis

    def test_coalesce_decoalesce_singletime(self):
        self.times = numpy.array([0.0])
        self.blockvis = create_blockvisibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                               weight=1.0, polarisation_frame=PolarisationFrame('stokesI'),
                                               channel_bandwidth=self.channel_bandwidth)
        # Fill in the vis values so each can be uniquely identified
        self.blockvis.data['vis'] = range(self.blockvis.nvis)
        cvis = coalesce_visibility(self.blockvis, time_coal=1.0)
        assert numpy.min(cvis.frequency) == numpy.min(self.frequency)
        assert numpy.min(cvis.frequency) > 0.0
        dvis = decoalesce_visibility(cvis)
        assert dvis.nvis == self.blockvis.nvis
    
    def test_coalesce_decoalesce_tbgrid_vis_null(self):
        cvis = coalesce_visibility(self.blockvis, time_coal=0.0)
        assert numpy.min(cvis.frequency) == numpy.min(self.frequency)
        assert numpy.min(cvis.frequency) > 0.0
        
    def test_coalesce_decoalesce_with_iter(self):
        for rows in vis_timeslice_iter(self.blockvis):
            visslice = create_visibility_from_rows(self.blockvis, rows)
            cvisslice = convert_blockvisibility_to_visibility(visslice)
            assert numpy.min(cvisslice.frequency) == numpy.min(self.frequency)
            assert numpy.min(cvisslice.frequency) > 0.0
            dvisslice = decoalesce_visibility(cvisslice)
            assert dvisslice.nvis == visslice.nvis


if __name__ == '__main__':
    unittest.main()
