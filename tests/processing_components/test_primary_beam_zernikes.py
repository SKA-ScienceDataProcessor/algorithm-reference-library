"""Unit tests for testing support


"""

import logging
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt

from data_models.polarisation import PolarisationFrame

from processing_components.image.operations import export_image_to_fits, show_image
from processing_components.imaging.base import create_image_from_visibility
from processing_components.imaging.primary_beams import create_pb, create_vp, create_vp_generic_numeric
from processing_components.simulation.configurations import create_named_configuration
from processing_components.visibility.base import create_visibility

log = logging.getLogger(__name__)


class TestPrimaryBeams(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
    
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
    
    def test_create_voltage_patterns_zernike(self):
        
        self.createVis(freq=1.4e9)
        cellsize = 8 * numpy.pi / 180.0 / 280
        model = create_image_from_visibility(self.vis, npixel=512, cellsize=cellsize, override_cellsize=False)
        plt.clf()
        fig, axs = plt.subplots(7, 7, gridspec_kw={'hspace': 0, 'wspace': 0})
        for noll in range(1,50):
            zernikes = [{'coeff':1.0,'noll':noll}]
            vp = create_vp_generic_numeric(model, pointingcentre=None, diameter=15.0, blockage=0.0, taper='gaussian',
                                        edge=0.03162278, zernikes=zernikes, padding=2, use_local=True)
            vp_data = vp.data
            vp.data = numpy.real(vp_data)
            export_image_to_fits(vp, "%s/test_voltage_pattern_real_%s_NOLL%d.fits" % (self.dir, 'MID_ZERNIKES', noll))
            row = (noll-1)//7
            col = (noll-1) - 7 * row
            ax = axs[row, col]
            ax.imshow(vp.data[0,0], vmax=0.1, vmin=-0.01)
            #ax.set_title('Noll %d' % noll)
            ax.axis('off')

        plt.savefig("zernikes.png")
        plt.show()

if __name__ == '__main__':
    unittest.main()
