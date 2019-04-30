""" Unit tests for pointing

"""

import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from processing_components.calibration.pointing import create_pointingtable_from_blockvisibility
from processing_components.imaging.primary_beams import create_vp
from processing_components.simulation.configurations import create_named_configuration
from processing_components.simulation.pointing import create_gaintable_from_pointingtable
from processing_components.simulation.testing_support import create_test_image, simulate_pointingtable
from processing_components.simulation.testing_support import create_test_skycomponents_from_s3
from processing_components.visibility.base import create_blockvisibility
from processing_library.image.operations import create_image

log = logging.getLogger(__name__)


class TestPointing(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        
        self.midcore = create_named_configuration('MID', rmax=300.0)
        self.dir = arl_path('test_results')
        self.times = numpy.linspace(-6, 6, 3) * numpy.pi / (12.0)
        
        self.frequency = numpy.array([1e9])
        self.channel_bandwidth = numpy.array([1e7])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_blockvisibility(self.midcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre, weight=1.0,
                                          polarisation_frame=PolarisationFrame('stokesI'))
        self.vis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_image(npixel=512, cellsize=0.0015, polarisation_frame=PolarisationFrame("stokesI"),
                                  frequency=self.frequency, channel_bandwidth=self.channel_bandwidth,
                                  phasecentre=self.phasecentre)
        
        self.vp = create_vp(self.model, 'LOW')
    
    def test_create_pointingtable(self):
        beam = create_test_image(cellsize=0.0015, phasecentre=self.vis.phasecentre,
                                 frequency=self.frequency)
        
        for telescope in ['MID', 'LOW', 'ASKAP']:
            vp = create_vp(beam, telescope)
            pt = create_pointingtable_from_blockvisibility(self.vis, vp)
            pt = simulate_pointingtable(pt, 0.1, static_pointing_error=0.01)
            assert pt.pointing.shape == (3, 63, 1, 1, 2), pt.pointing.shape
    
    def test_create_gaintable_from_pointingtable(self):
        s3_components = [create_test_skycomponents_from_s3(flux_limit=5.0,
                                                           phasecentre=self.phasecentre,
                                                           frequency=self.frequency,
                                                           polarisation_frame=PolarisationFrame('stokesI'),
                                                           radius=0.2)[0]]
        
        pt = create_pointingtable_from_blockvisibility(self.vis)
        pt = simulate_pointingtable(pt, pointing_error=0.01, static_pointing_error=0.001)
        vp = create_vp(self.model, 'MID')
        gt = create_gaintable_from_pointingtable(self.vis, s3_components, pt, vp)
        assert gt[0].gain.shape == (3, 63, 1, 1, 1), gt[0].gain.shape
    
    def test_create_gaintable_from_pointingtable_global(self):
        s3_components = [create_test_skycomponents_from_s3(flux_limit=5.0,
                                                           phasecentre=self.phasecentre,
                                                           frequency=self.frequency,
                                                           polarisation_frame=PolarisationFrame('stokesI'),
                                                           radius=0.2)[0]]
    
        pt = create_pointingtable_from_blockvisibility(self.vis)
        pt = simulate_pointingtable(pt, pointing_error=0.0, static_pointing_error=0.0,
                                    global_pointing_error=[0.0001, 0.0001])
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(pt.pointing[...,0].flat, pt.pointing[...,1].flat, '.')
        plt.show()
        vp = create_vp(self.model, 'MID')
        gt = create_gaintable_from_pointingtable(self.vis, s3_components, pt, vp)
        assert gt[0].gain.shape == (3, 63, 1, 1, 1), gt[0].gain.shape
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(numpy.real(gt[0].gain.flat), numpy.imag(gt[0].gain.flat), '.')
        plt.show()