""" Unit tests for pointing

"""

import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from arl.data_models.polarisation import PolarisationFrame

from arl.processing_components.skycomponent.operations import create_skycomponent
from arl.processing_components.calibration.pointing import create_pointingtable_from_blockvisibility
from arl.processing_components.imaging.primary_beams import create_vp
from arl.processing_components.simulation import create_named_configuration
from arl.processing_components.simulation.pointing import simulate_gaintable_from_pointingtable
from arl.processing_components.simulation import create_test_image, simulate_pointingtable
from arl.processing_components.simulation import create_test_skycomponents_from_s3
from arl.processing_components.visibility.base import create_blockvisibility
from arl.processing_library.image.operations import create_image

log = logging.getLogger(__name__)


class TestPointing(unittest.TestCase):
    def setUp(self):
        from arl.data_models.parameters import arl_path
        
        self.doplot = True
        
        self.midcore = create_named_configuration('MID', rmax=300.0)
        self.nants = len(self.midcore.names)
        self.dir = arl_path('test_results')
        self.ntimes = 301
        self.times = numpy.linspace(-6.0, 6.0, self.ntimes) * numpy.pi / (12.0)
        
        self.frequency = numpy.array([1.4e9])
        self.channel_bandwidth = numpy.array([1e7])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-50.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_blockvisibility(self.midcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre, weight=1.0,
                                          polarisation_frame=PolarisationFrame('stokesI'))
        self.vis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_image(npixel=2048, cellsize=0.0003, polarisation_frame=PolarisationFrame("stokesI"),
                                  frequency=self.frequency, channel_bandwidth=self.channel_bandwidth,
                                  phasecentre=self.phasecentre)
    
    def test_create_gaintable_from_pointingtable_circlecut(self):
        self.sidelobe = SkyCoord(ra=+15.0 * u.deg, dec=-49.4 * u.deg, frame='icrs', equinox='J2000')
        comp = create_skycomponent(direction=self.sidelobe, flux=[[1.0]], frequency=self.frequency,
                                   polarisation_frame=PolarisationFrame('stokesI'))
    
        telescopes = ['MID', 'MID_GAUSS', 'MID_GRASP']
        for telescope in telescopes:
            pt = create_pointingtable_from_blockvisibility(self.vis)
            pt = simulate_pointingtable(pt, pointing_error=0.0,
                                        global_pointing_error=[0.0, 0.0])
            vp = create_vp(self.model, telescope)
            gt = simulate_gaintable_from_pointingtable(self.vis, [comp], pt, vp)
            if self.doplot:
                import matplotlib.pyplot as plt
                plt.clf()
                plt.plot(gt[0].time, numpy.real(1.0 / gt[0].gain[:, 0, 0, 0, 0]), '.', label='Real')
                plt.plot(gt[0].time, numpy.imag(1.0 / gt[0].gain[:, 0, 0, 0, 0]), '.', label='Imaginary')
                plt.legend()
                plt.xlabel('Time (s)')
                plt.ylabel('Gain')
                plt.title('test_create_gaintable_from_pointingtable_%s' % telescope)
                plt.show()
            assert gt[0].gain.shape == (self.ntimes, self.nants, 1, 1, 1), gt[0].gain.shape


if __name__ == '__main__':
    unittest.main()
