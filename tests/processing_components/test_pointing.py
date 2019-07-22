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
from processing_components.simulation.pointing import simulate_gaintable_from_pointingtable
from processing_components.simulation.testing_support import create_test_image, simulate_pointingtable
from processing_components.simulation.testing_support import create_test_skycomponents_from_s3
from processing_components.skycomponent.operations import create_skycomponent
from processing_components.visibility.base import create_blockvisibility
from processing_library.image.operations import create_image

log = logging.getLogger(__name__)


class TestPointing(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        
        self.doplot = True
        
        self.midcore = create_named_configuration('MID', rmax=300.0)
        self.nants = len(self.midcore.names)
        self.dir = arl_path('test_results')
        self.ntimes = 300
        self.times = numpy.linspace(-12.0, 12.0, self.ntimes) * numpy.pi / (12.0)
        
        self.frequency = numpy.array([1e9])
        self.channel_bandwidth = numpy.array([1e7])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_blockvisibility(self.midcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre, weight=1.0,
                                          polarisation_frame=PolarisationFrame('stokesI'))
        self.vis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_image(npixel=512, cellsize=0.00015, polarisation_frame=PolarisationFrame("stokesI"),
                                  frequency=self.frequency, channel_bandwidth=self.channel_bandwidth,
                                  phasecentre=self.phasecentre)
    
    def test_create_pointingtable(self):
        beam = create_test_image(cellsize=0.0015, phasecentre=self.vis.phasecentre,
                                 frequency=self.frequency)
        
        for telescope in ['MID', 'LOW', 'ASKAP']:
            vp = create_vp(beam, telescope)
            pt = create_pointingtable_from_blockvisibility(self.vis, vp)
            pt = simulate_pointingtable(pt, 0.1, static_pointing_error=[0.01, 0.001])
            assert pt.pointing.shape == (self.ntimes, self.nants, 1, 1, 2), pt.pointing.shape
    
    def test_create_gaintable_from_pointingtable(self):
        s3_components = create_test_skycomponents_from_s3(flux_limit=5.0,
                                                          phasecentre=self.phasecentre,
                                                          frequency=self.frequency,
                                                          polarisation_frame=PolarisationFrame('stokesI'),
                                                          radius=0.2)
        
        pt = create_pointingtable_from_blockvisibility(self.vis)
        pt = simulate_pointingtable(pt, pointing_error=0.01, static_pointing_error=[0.001,0.0001])
        vp = create_vp(self.model, 'MID')
        gt = simulate_gaintable_from_pointingtable(self.vis, s3_components, pt, vp)
        assert gt[0].gain.shape == (self.ntimes, self.nants, 1, 1, 1), gt[0].gain.shape
    
    def test_create_gaintable_from_pointingtable_dynamic(self):
        comp = create_skycomponent(direction=self.phasecentre, flux=[[1.0]], frequency=self.frequency,
                                   polarisation_frame=PolarisationFrame('stokesI'))
        
        pt = create_pointingtable_from_blockvisibility(self.vis)
        pt = simulate_pointingtable(pt, pointing_error=0.01, static_pointing_error=None,
                                    global_pointing_error=[0.0, 0.0])
        vp = create_vp(self.model, 'MID')
        gt = simulate_gaintable_from_pointingtable(self.vis, [comp], pt, vp)
        if self.doplot:
            import matplotlib.pyplot as plt
            plt.clf()
            plt.plot(gt[0].time, numpy.real(1.0 / gt[0].gain[:, 0, 0, 0, 0]), '.')
            plt.plot(gt[0].time, numpy.imag(1.0 / gt[0].gain[:, 0, 0, 0, 0]), '.')
            plt.title('test_create_gaintable_from_pointingtable_dynamic')
            plt.show()
        assert gt[0].gain.shape == (self.ntimes, self.nants, 1, 1, 1), gt[0].gain.shape
    
    def test_create_gaintable_from_pointingtable_dynamic_radec(self):
        comp = create_skycomponent(direction=self.phasecentre, flux=[[1.0]], frequency=self.frequency,
                                   polarisation_frame=PolarisationFrame('stokesI'))
        
        pt = create_pointingtable_from_blockvisibility(self.vis)
        pt = simulate_pointingtable(pt, pointing_error=0.01, static_pointing_error=None,
                                    global_pointing_error=[0.0, 0.0])
        vp = create_vp(self.model, 'MID', use_local=False)
        gt = simulate_gaintable_from_pointingtable(self.vis, [comp], pt, vp, use_radec=True)
        if self.doplot:
            import matplotlib.pyplot as plt
            plt.clf()
            plt.plot(gt[0].time, numpy.real(1.0 / gt[0].gain[:, 0, 0, 0, 0]), '.')
            plt.plot(gt[0].time, numpy.imag(1.0 / gt[0].gain[:, 0, 0, 0, 0]), '.')
            plt.title('test_create_gaintable_from_pointingtable_dynamic_radec')
            plt.show()
        assert gt[0].gain.shape == (self.ntimes, self.nants, 1, 1, 1), gt[0].gain.shape
    
    def test_create_gaintable_from_pointingtable_static(self):
        comp = create_skycomponent(direction=self.phasecentre, flux=[[1.0]], frequency=self.frequency,
                                   polarisation_frame=PolarisationFrame('stokesI'))
        
        pt = create_pointingtable_from_blockvisibility(self.vis)
        pt = simulate_pointingtable(pt, pointing_error=0.0, static_pointing_error=[0.01, 0.001],
                                    global_pointing_error=[0.0, 0.0])
        vp = create_vp(self.model, 'MID')
        gt = simulate_gaintable_from_pointingtable(self.vis, [comp], pt, vp)
        if self.doplot:
            import matplotlib.pyplot as plt
            plt.clf()
            plt.plot(gt[0].time, numpy.real(1.0 / gt[0].gain[:, 0, 0, 0, 0]), '.')
            plt.plot(gt[0].time, numpy.imag(1.0 / gt[0].gain[:, 0, 0, 0, 0]), '.')
            plt.title('test_create_gaintable_from_pointingtable_static')
            plt.show()
        
        assert gt[0].gain.shape == (self.ntimes, self.nants, 1, 1, 1), gt[0].gain.shape
    
    def test_create_gaintable_from_pointingtable_dynamic_static(self):
        comp = create_skycomponent(direction=self.phasecentre, flux=[[1.0]], frequency=self.frequency,
                                   polarisation_frame=PolarisationFrame('stokesI'))
        
        pt = create_pointingtable_from_blockvisibility(self.vis)
        pt = simulate_pointingtable(pt, pointing_error=0.01, static_pointing_error=[0.01, 0.001],
                                    global_pointing_error=[0.0, 0.0])
        vp = create_vp(self.model, 'MID')
        gt = simulate_gaintable_from_pointingtable(self.vis, [comp], pt, vp)
        if self.doplot:
            import matplotlib.pyplot as plt
            plt.clf()
            plt.plot(gt[0].time, numpy.real(1.0 / gt[0].gain[:, 0, 0, 0, 0]), '.')
            plt.plot(gt[0].time, numpy.imag(1.0 / gt[0].gain[:, 0, 0, 0, 0]), '.')
            plt.title('test_create_gaintable_from_pointingtable_dynamic_static')
            plt.show()
        assert gt[0].gain.shape == (self.ntimes, self.nants, 1, 1, 1), gt[0].gain.shape
    
    def test_create_gaintable_from_pointingtable_global(self):
        
        comp = create_skycomponent(direction=self.phasecentre, flux=[[1.0]], frequency=self.frequency,
                                   polarisation_frame=PolarisationFrame('stokesI'))
        
        import matplotlib.pyplot as plt
        pt = create_pointingtable_from_blockvisibility(self.vis)
        pt = simulate_pointingtable(pt, pointing_error=0.0, static_pointing_error=None,
                                    global_pointing_error=[0.0, 0.01])
        vp = create_vp(self.model, 'MID')
        gt = simulate_gaintable_from_pointingtable(self.vis, [comp], pt, vp)
        if self.doplot:
            plt.clf()
            plt.plot(gt[0].time, numpy.real(1.0 / gt[0].gain[:, 0, 0, 0, 0]), '.')
            plt.plot(gt[0].time, numpy.imag(1.0 / gt[0].gain[:, 0, 0, 0, 0]), '.')
            plt.title('test_create_gaintable_from_pointingtable_global')
            plt.show()
        assert gt[0].gain.shape == (self.ntimes, self.nants, 1, 1, 1), gt[0].gain.shape
    
    def test_create_gaintable_from_pointingtable_global_dynamic(self):
        comp = create_skycomponent(direction=self.phasecentre, flux=[[1.0]], frequency=self.frequency,
                                   polarisation_frame=PolarisationFrame('stokesI'))
        
        pt = create_pointingtable_from_blockvisibility(self.vis)
        pt = simulate_pointingtable(pt, pointing_error=0.01, static_pointing_error=None,
                                    global_pointing_error=[0.0, 0.01])
        vp = create_vp(self.model, 'MID')
        gt = simulate_gaintable_from_pointingtable(self.vis, [comp], pt, vp)
        if self.doplot:
            import matplotlib.pyplot as plt
            plt.clf()
            plt.plot(gt[0].time, numpy.real(1.0 / gt[0].gain[:, 0, 0, 0, 0]), '.')
            plt.plot(gt[0].time, numpy.imag(1.0 / gt[0].gain[:, 0, 0, 0, 0]), '.')
            plt.title('test_create_gaintable_from_pointingtable_global_dynamic')
            plt.show()
        assert gt[0].gain.shape == (self.ntimes, self.nants, 1, 1, 1), gt[0].gain.shape
    
    def test_create_gaintable_from_pointingtable_GRASP(self):
        comp = create_skycomponent(direction=self.phasecentre, flux=[[1.0]], frequency=self.frequency,
                                   polarisation_frame=PolarisationFrame('stokesI'))
        
        pt = create_pointingtable_from_blockvisibility(self.vis)
        pt = simulate_pointingtable(pt, pointing_error=0.0, static_pointing_error=None,
                                    global_pointing_error=[0.0, 0.01])
        vp = create_vp(self.model, 'MID_GRASP')
        gt = simulate_gaintable_from_pointingtable(self.vis, [comp], pt, vp)
        if self.doplot:
            import matplotlib.pyplot as plt
            plt.clf()
            plt.plot(gt[0].time, numpy.real(1.0 / gt[0].gain[:, 0, 0, 0, 0]), '.')
            plt.plot(gt[0].time, numpy.imag(1.0 / gt[0].gain[:, 0, 0, 0, 0]), '.')
            plt.title('test_create_gaintable_from_pointingtable_global_dynamic')
            plt.show()
        assert gt[0].gain.shape == (self.ntimes, self.nants, 1, 1, 1), gt[0].gain.shape


if __name__ == '__main__':
    unittest.main()
