import sys
import unittest

from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.fourier_transforms.convolutional_gridding import *
from arl.fourier_transforms.variable_kernels import *
from arl.util.testing_support import create_named_configuration
from arl.visibility.operations import create_visibility

log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

log = logging.getLogger("tests.test_convolutional_gridding_kernel")


class TestConvolutionalGriddingVariable(unittest.TestCase):
    def setUp(self):
        self.params = {'npixel': 512,
                       'cellsize': 0.001,
                       'spectral_mode': 'channel',
                       'channelwidth': 5e7,
                       'reffrequency': 1e8,
                       'image_partitions': 5,
                       'padding': 1,
                       'kernel': 'transform',
                       'oversampling': 8}
        
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = numpy.arange(-numpy.pi / 4.0, +numpy.pi / 4.0, 0.25)
        self.frequency = numpy.array([1e8])
        
        self.reffrequency = numpy.max(self.frequency)
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                     phasecentre=self.phasecentre)


    def assertAlmostEqualScalar(self, a, result=1.0):
        w = result * numpy.ones_like(result)
    
    
    def _test_pattern(self, npixel):
        return coordinates2(npixel)[0] + coordinates2(npixel)[1] * 1j
    
    
    def test_standard_kernel_lambda(self):
        shape = (self.params['npixel'], self.params['npixel'])
        skl = standard_kernel_lambda(self.vis, shape)
        assert len(skl(10, 0).shape) == 4, "Shape is wrong: %s" % (skl(10).shape)

    
    
    def test_w_kernel_lambda(self):
        shape = (self.params['npixel'], self.params['npixel'])
        wkl = w_kernel_lambda(self.vis, shape, fov=0.1)[0]
        assert len(wkl(10,0).shape) == 4, "Shape is wrong: %s" % (wkl(10,0).shape)
        for row in range(len(self.vis.data)):
            assert len(wkl(row, 0).shape) == 4


if __name__ == '__main__':
    unittest.main()
