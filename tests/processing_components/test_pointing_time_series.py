""" Unit tests for pointing

"""

import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.parameters import arl_path
from data_models.polarisation import PolarisationFrame
from data_models.memory_data_models import Skycomponent

from processing_components.skycomponent.operations import create_skycomponent
from processing_components.calibration.pointing import create_pointingtable_from_blockvisibility
from processing_components.imaging.primary_beams import create_vp
from processing_components.simulation.configurations import create_named_configuration
from processing_components.simulation.pointing import create_gaintable_from_pointingtable
from processing_components.simulation.testing_support import create_test_image, simulate_pointingtable, \
    simulate_pointingtable_from_timeseries
from processing_components.simulation.testing_support import create_test_skycomponents_from_s3
from processing_components.visibility.base import create_blockvisibility
from processing_library.image.operations import create_image

log = logging.getLogger(__name__)


class TestPointing(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        
        self.doplot = True
        
        self.midcore = create_named_configuration('MID', rmax=100.0)
        self.nants = len(self.midcore.names)
        self.dir = arl_path('test_results')
        self.ntimes = 203
        self.times = numpy.linspace(-0.28194, 0.28194, self.ntimes)
        
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
    
    def test_simulate_gaintable_from_time_series(self):
        numpy.random.seed(18051955)
        offset_phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-44.58 * u.deg, frame='icrs', equinox='J2000')
        component = [Skycomponent(frequency=self.frequency, direction=offset_phasecentre,
                                      polarisation_frame=PolarisationFrame("stokesI"), flux=[[1.0]])]
        
        pt = create_pointingtable_from_blockvisibility(self.vis)
        pt = simulate_pointingtable_from_timeseries(pt, scaling=1.0)
        
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(pt.time, pt.pointing[:,0,0,0,0], '.')
        plt.plot(pt.time, pt.pointing[:,0,0,0,1], '.')
        plt.show()

        vp = create_vp(self.model, 'MID')
        gt = create_gaintable_from_pointingtable(self.vis, component, pt, vp)
        assert gt[0].gain.shape == (self.ntimes, self.nants, 1, 1, 1), gt[0].gain.shape
        
        plt.clf()
        plt.plot(gt[0].time, 1.0/numpy.real(gt[0].gain[:,0,0,0,0]), '.')
#        plt.plot(gt[0].time, numpy.imag(gt[0].gain[:,0,0,0,0]), '.')
        plt.show()



if __name__ == '__main__':
    unittest.main()
