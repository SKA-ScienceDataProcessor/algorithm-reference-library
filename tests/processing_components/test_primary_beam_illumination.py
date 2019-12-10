"""Unit tests for testing support


"""

import logging
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt

from arl.data_models.polarisation import PolarisationFrame

from arl.processing_components.image.operations import export_image_to_fits, show_image
from arl.processing_components.imaging.base import create_image_from_visibility
from arl.processing_components.imaging.primary_beams import create_pb, create_vp, create_vp_generic_numeric
from arl.processing_components.simulation import create_named_configuration
from arl.processing_components.visibility.base import create_visibility

log = logging.getLogger(__name__)


class TestPrimaryBeams(unittest.TestCase):
    def setUp(self):
        from arl.data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.persist = False
    
    def createVis(self, config='MID', dec=-35.0, rmax=1e3, freq=1e9):
        self.frequency = numpy.linspace(freq, 1.5 * freq, 3)
        self.channel_bandwidth = numpy.array([2.5e7, 2.5e7, 2.5e7])
        self.flux = numpy.array([[100.0], [100.0], [100.0]])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.config = create_named_configuration(config)
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        nants = self.config.xyz.shape[0]
        assert nants > 1
        assert len(self.config.names) == nants
        assert len(self.config.mount) == nants
        
        self.config = create_named_configuration(config, rmax=rmax)
        self.phasecentre = SkyCoord(ra=+15 * u.deg, dec=dec * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_visibility(self.config, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('stokesI'))
    
    def test_create_voltage_patterns_illumination(self):
        
        self.createVis(freq=1.4e9)
        cellsize = 8 * numpy.pi / 180.0 / 280
        model = create_image_from_visibility(self.vis, npixel=512, cellsize=cellsize, override_cellsize=False)
        plt.clf()
        fig, axs = plt.subplots(5, 5, gridspec_kw={'hspace': 0, 'wspace': 0})
        # (r ** 2 + rho * (dx * dy) + diff * (dx ** 2 - dy ** 2))
        for irho, rho in enumerate([-0.1, -0.05, 0.0, 0.05, 0.1]):
            for idiff, diff in enumerate([-0.2, -0.15, -0.1, -0.05, 0.0]):
                vp = create_vp_generic_numeric(model, pointingcentre=None, diameter=15.0, blockage=0.0, taper='gaussian',
                                            edge=0.03162278, padding=2, use_local=True, rho=rho, diff=diff)
                vp_data = vp.data
                vp.data = numpy.real(vp_data)
                if self.persist: export_image_to_fits(vp, "%s/test_voltage_pattern_real_%s_rho%.3f_diff%.3f.fits" %
                                     (self.dir, "MID_TAPER", rho, diff))
                ax = axs[irho, idiff]
                ax.imshow(vp.data[0,0])#, vmax=0.1, vmin=-0.01)
                ax.axis('off')

        plt.show()

if __name__ == '__main__':
    unittest.main()
