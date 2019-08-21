""" Unit tests for surface simulations

"""

import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.memory_data_models import Skycomponent
from data_models.polarisation import PolarisationFrame
from processing_components.imaging.primary_beams import create_vp_generic_numeric, create_vp
from processing_components.simulation.configurations import create_named_configuration
from processing_components.simulation.surface import simulate_gaintable_from_voltage_patterns
from processing_components.visibility.base import create_blockvisibility
from processing_library.image.operations import create_image

log = logging.getLogger(__name__)


class TestSurface(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        
        self.doplot = True
        
        self.midcore = create_named_configuration('MID', rmax=100.0)
        self.nants = len(self.midcore.names)
        self.dir = arl_path('test_results')
        self.ntimes = 100
        interval = 10.0
        self.times = numpy.arange(0.0, float(self.ntimes)) * interval
        self.times *= numpy.pi / 43200.0
        
        self.frequency = numpy.array([1.4e9])
        self.channel_bandwidth = numpy.array([1e7])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_blockvisibility(self.midcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre, weight=1.0,
                                          polarisation_frame=PolarisationFrame('stokesI'))
        self.vis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_image(npixel=512, cellsize=0.001, polarisation_frame=PolarisationFrame("stokesI"),
                                  frequency=self.frequency, channel_bandwidth=self.channel_bandwidth,
                                  phasecentre=self.phasecentre)
    
    def test_simulate_gaintable_from_voltage_patterns(self):
        numpy.random.seed(18051955)
        offset_phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-44.58 * u.deg, frame='icrs', equinox='J2000')
        component = [Skycomponent(frequency=self.frequency, direction=offset_phasecentre,
                                  polarisation_frame=PolarisationFrame("stokesI"), flux=[[1.0]])]

        key_nolls = [3, 5, 6, 7]
        vp_list = list()
        vp_list.append(create_vp(self.model, 'MID_GAUSS', use_local=True))
        vp_coeffs = numpy.ones([self.nants, len(key_nolls)+1])
        for inoll, noll in enumerate(key_nolls):
            zernike = {'coeff': 1.0, 'noll': noll}
            vp_coeffs[:, inoll+1] = numpy.random.normal(0.0, 0.03, self.nants)
            vp_list.append(create_vp_generic_numeric(self.model, pointingcentre=None, diameter=15.0, blockage=0.0,
                                                      taper='gaussian',
                                                      edge=0.03162278, zernikes=[zernike], padding=2, use_local=True))

        gt = simulate_gaintable_from_voltage_patterns(self.vis, component, vp_list, vp_coeffs)

        import matplotlib.pyplot as plt
        assert gt[0].gain.shape == (self.ntimes, self.nants, 1, 1, 1), gt[0].gain.shape
        
        plt.clf()
        for ant in range(self.nants):
            plt.plot(gt[0].time, 1.0 / numpy.real(gt[0].gain[:, ant, 0, 0, 0]), '.')
        plt.xlabel('Time (s)')
        plt.ylabel('Gain')
        plt.show()


if __name__ == '__main__':
    unittest.main()
