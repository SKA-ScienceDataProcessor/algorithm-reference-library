"""Unit tests for Fourier transforms

realtimcornwell@gmail.com
"""
import unittest

from arl.util.testing_support import create_named_configuration, run_unittests
from arl.visibility.operations import create_visibility
from arl.visibility.compress import *

log = logging.getLogger(__name__)


class TestCompress(unittest.TestCase):

    def setUp(self):
        self.params = {'npixel': 128,
                       'npol': 1,
                       'cellsize': 0.0018,
                       'reffrequency': 1e8}
    
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = numpy.arange(- numpy.pi / 4.0, 1.001 * numpy.pi / 4.0, numpy.pi / 16.0)
        self.frequency = numpy.array([1e8])
    
        self.reffrequency = numpy.max(self.frequency)
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox=2000.0)
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre)
        self.model = create_image_from_visibility(self.vis, **self.params)

    def test_compress_decompress_grid_vis(self):
        """Test compression"""
        cvis = compress_visibility(self.vis, self.model, compression='uvgrid')
        dvis = decompress_visibility(cvis, self.vis, self.model, compression='uvgrid')
        assert dvis.nvis == self.vis.nvis
        
if __name__ == '__main__':
    run_unittests()
