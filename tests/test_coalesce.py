"""Unit tests for Fourier transforms

realtimcornwell@gmail.com
"""
import unittest

from arl.util.run_unittests import run_unittests
from arl.util.testing_support import create_named_configuration
from arl.visibility.coalesce import *
from arl.visibility.operations import create_blockvisibility

log = logging.getLogger(__name__)


class TestCoalesce(unittest.TestCase):
    def setUp(self):

        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 30 * 3.76, 3.76)
        df = 27343.75000
        self.frequency = numpy.array([1e8-df, 1e8, 1e8+df])
        self.channel_bandwidth = numpy.array([27343.75, 27343.75, 27343.75])
        self.phasecentre = SkyCoord(ra=+0.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.blockvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                               channel_bandwidth=self.channel_bandwidth,
                                               phasecentre=self.phasecentre, weight=1.0,
                                               polarisation_frame=PolarisationFrame('stokesI'))
    
    def test_coalesce_decoalesce(self):
        cvis, cindex = coalesce_visibility(self.blockvis, coalescence_factor=1.0)
        assert numpy.min(cvis.frequency) == numpy.min(self.frequency)
        assert numpy.min(cvis.frequency) > 0.0
        dvis = decoalesce_visibility(cvis, self.blockvis, cindex=cindex)
        assert dvis.nvis == self.blockvis.nvis
        dvis = decoalesce_visibility(cvis, self.blockvis, cindex=cindex, overwrite=True)
        assert dvis.nvis == self.blockvis.nvis
    
    def test_coalesce_decoalesce_singletime(self):
        self.times = numpy.array([0.0])
        self.blockvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                               channel_bandwidth=self.channel_bandwidth,
                                               phasecentre=self.phasecentre, weight=1.0,
                                               polarisation_frame=PolarisationFrame('stokesI'))
        # Fill in the vis values so each can be uniquely identified
        self.blockvis.data['vis'] = range(self.blockvis.nvis)
        cvis, cindex = coalesce_visibility(self.blockvis, coalescence_factor=1.0)
        assert numpy.min(cvis.frequency) == numpy.min(self.frequency)
        assert numpy.min(cvis.frequency) > 0.0
        dvis = decoalesce_visibility(cvis, self.blockvis, cindex=cindex)
        assert dvis.nvis == self.blockvis.nvis
    
    def test_coalesce_decoalesce_tbgrid_vis_null(self):
        cvis, cindex = coalesce_visibility(self.blockvis, coalescence_factor=0.0)
        assert numpy.min(cvis.frequency) == numpy.min(self.frequency)
        assert numpy.min(cvis.frequency) > 0.0

if __name__ == '__main__':
    unittest.main()
