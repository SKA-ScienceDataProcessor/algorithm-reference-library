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
                       'cellsize': 0.001,
                       'reffrequency': 1e8}
    
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 600.0, 30.0)
        self.frequency = numpy.array([1e8])
    
        self.reffrequency = numpy.max(self.frequency)
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox=2000.0)
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                     weight=1.0, npol=1)
        self.vis.data['vis'][:,0,0] = self.vis.data['time']
        self.model = create_image_from_visibility(self.vis, **self.params)

    def test_compress_decompress_uvgrid_vis(self):
        cvis, _ = compress_visibility(self.vis, self.model, compression='uv')
        dvis = decompress_visibility(cvis, self.vis, self.model, compression='uv')
        assert dvis.nvis == self.vis.nvis


    def test_compress_decompress_tbgrid_vis(self):
        cvis, cindex = compress_visibility(self.vis, self.model, compression='tb')
        numpy.testing.assert_array_equal(cvis.vis[:,0,0].real, cvis.time)
        dvis = decompress_visibility(cvis, self.vis, cindex=cindex, compression='tb')
        numpy.testing.assert_array_equal(self.vis.time, dvis.time)
        numpy.testing.assert_array_equal(self.vis.antenna1, dvis.antenna1)
        numpy.testing.assert_array_equal(self.vis.antenna2, dvis.antenna2)
        assert dvis.nvis == self.vis.nvis
        
    def test_average_chunks(self):
        
        arr = numpy.linspace(0.0, 100.0, 11)
        wts = numpy.ones_like(arr)
        carr, cwts = average_chunks(arr, wts, 2)
        assert len(carr) == len(cwts)
        answerarr = numpy.array([5., 25., 45., 65.0, 85.0, 100.0])
        answerwts = numpy.array([2.0, 2.0, 2.0, 2.0, 2.0, 1.0])
        numpy.testing.assert_array_equal(carr, answerarr)
        numpy.testing.assert_array_equal(cwts, answerwts)

if __name__ == '__main__':
    run_unittests()
