"""Unit tests for Fourier transforms

realtimcornwell@gmail.com
"""
import unittest

from arl.util.testing_support import create_named_configuration, run_unittests
from arl.visibility.operations import create_visibility
from arl.fourier_transforms.ftprocessor_base import create_image_from_visibility
from arl.visibility.compress import *

log = logging.getLogger(__name__)


class TestCompress(unittest.TestCase):

    def setUp(self):
    
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, 5)
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox=2000.0)
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                               phasecentre=self.phasecentre, weight=1.0, npol=1)
        # Fill in the vis values so each can be uniquely identified
        self.vis.data['vis'] = range(self.vis.nvis)
 
    def test_compress_decompress_tbgrid_vis(self):
        cvis, cindex = compress_visibility(self.vis, compression_factor=1.0)
        dvis = decompress_visibility(cvis, self.vis, cindex=cindex)
        assert dvis.nvis == self.vis.nvis

    def test_compress_decompress_tbgrid_vis_null(self):
        cvis, cindex = compress_visibility(self.vis, compression_factor=0.0)
        assert cindex is None
if __name__ == '__main__':
    run_unittests()
