"""Unit tests for Fourier transform processors


"""
import logging
import unittest
import copy

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.data.polarisation import PolarisationFrame
from arl.imaging.base import create_image_from_visibility
from arl.imaging.weighting import weight_visibility, taper_visibility_Gaussian, \
    taper_visibility_tukey
from arl.util.testing_support import create_named_configuration
from arl.visibility.base import create_visibility

log = logging.getLogger(__name__)


class TestWeighting(unittest.TestCase):
    def setUp(self):
        import os
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        self.params = {'npixel': 512,
                       'nchan': 1,
                       'reffrequency': 1e8,
                       'facets': 8,
                       'padding': 2,
                       'oversampling': 2,
                       'timeslice': 1000.0}
    
    def actualSetUp(self, time=None, frequency=None, dospectral=False, dopol=False):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 5)
        
        if time is not None:
            self.times = time
        log.info("Times are %s" % (self.times))
        
        if dospectral:
            self.nchan=3
            self.frequency = numpy.array([0.9e8, 1e8, 1.1e8])
            self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        else:
            self.frequency = numpy.array([1e8])
            self.channel_bandwidth = numpy.array([1e7])
            
        if dopol:
            self.vis_pol = PolarisationFrame('linear')
            self.image_pol = PolarisationFrame('stokesIQUV')
        else:
            self.vis_pol = PolarisationFrame('stokesI')
            self.image_pol = PolarisationFrame('stokesI')

        if dopol:
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        else:
            f = numpy.array([100.0])

        if dospectral:
            flux = numpy.array([f, 0.8 * f, 0.6 * f])
        else:
            flux = numpy.array([f])


        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency,
                                              channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                              weight=1.0, polarisation_frame=self.vis_pol)
        self.uvw = self.componentvis.data['uvw']
        self.componentvis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_image_from_visibility(self.componentvis, npixel=512, cellsize=0.001,
                                                  nchan=len(self.frequency),
                                                  polarisation_frame=self.image_pol)

    def test_weighting(self):
        self.actualSetUp()
        vis, density, densitygrid = weight_visibility(self.componentvis, self.model, weighting='uniform')
        assert vis.nvis == self.componentvis.nvis
        assert len(density) == vis.nvis
        assert numpy.std(vis.imaging_weight) > 0.0
        assert densitygrid.data.shape == self.model.data.shape
        vis, density, densitygrid = weight_visibility(self.componentvis, self.model, weighting='natural')
        assert density is None
        assert densitygrid is None

    def test_tapering_Gaussian(self):
        self.actualSetUp()
        original_weight = copy.deepcopy(self.componentvis.imaging_weight)
        vis = taper_visibility_Gaussian(self.componentvis, algorithm='Gaussian', beam=0.1)
        assert vis.nvis == self.componentvis.nvis

    def test_tapering_Tukey(self):
        self.actualSetUp()
        original_weight = copy.deepcopy(self.componentvis.imaging_weight)
        vis = taper_visibility_tukey(self.componentvis, algorithm='Tukey', tukey=0.25)
        assert vis.nvis == self.componentvis.nvis


if __name__ == '__main__':
    unittest.main()
