""" Unit tests for Fourier transform processors


"""
import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame

from processing_components.imaging.base import create_image_from_visibility
from processing_components.simulation.testing_support import ingest_unittest_visibility, create_unittest_model
from processing_components.simulation.configurations import create_named_configuration

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImagingFunctions(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
    
    def actualSetUp(self, add_errors=False, freqwin=1, block=False, dospectral=True, dopol=False):
        
        self.npixel = 256
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = freqwin
        self.vis_list = list()
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        if freqwin > 1:
            self.channelwidth = numpy.array(freqwin * [self.frequency[1] - self.frequency[0]])
        else:
            self.channelwidth = numpy.array([1e6])
        
        if dopol:
            self.vis_pol = PolarisationFrame('linear')
            self.image_pol = PolarisationFrame('stokesIQUV')
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        else:
            self.vis_pol = PolarisationFrame('stokesI')
            self.image_pol = PolarisationFrame('stokesI')
            f = numpy.array([100.0])
        
        if dospectral:
            flux = numpy.array([f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency])
        else:
            flux = numpy.array([f])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = ingest_unittest_visibility(self.low, self.frequency, self.channelwidth, self.times,
                                              self.vis_pol, self.phasecentre, block=block)
        
        self.model = create_unittest_model(self.vis, self.image_pol, npixel=self.npixel)
    
    def test_create_image_from_visibility(self):
        self.actualSetUp()
        im = create_image_from_visibility(self.vis, nchan=1, npixel=128)
        assert im.data.shape == (1, 1, 128, 128)
        im = create_image_from_visibility(self.vis, frequency=self.frequency, npixel=128)
        assert im.data.shape == (len(self.frequency), 1, 128, 128)
        im = create_image_from_visibility(self.vis, frequency=self.frequency, npixel=128,
                                          nchan=1)
        assert im.data.shape == (1, 1, 128, 128)


if __name__ == '__main__':
    unittest.main()
