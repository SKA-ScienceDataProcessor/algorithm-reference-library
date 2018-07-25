""" Unit tests for skycomponents

"""

import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame

from processing_components.image.operations import export_image_to_fits
from workflows.serial.imaging.imaging_serial import predict_function, invert_function
from processing_components.imaging.base import predict_skycomponent_visibility
from processing_components.skycomponent.operations import insert_skycomponent, create_skycomponent
from processing_components.simulation.testing_support import create_test_image, create_named_configuration
from processing_components.visibility.base import create_visibility

log = logging.getLogger(__name__)


class TestSkycomponentInsert(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.dir = arl_path('test_results')
        self.times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 7)
        self.image_frequency = numpy.linspace(0.9e8, 1.1e8, 5)
        self.component_frequency = numpy.linspace(0.8e8, 1.2e8, 7)
        self.channel_bandwidth = numpy.array(5*[1e7])
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_visibility(self.lowcore, self.times, self.image_frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('stokesI'))
        self.vis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_test_image(cellsize=0.0015, phasecentre=self.vis.phasecentre, frequency=self.image_frequency)
        self.model.data[self.model.data > 1.0] = 1.0
        self.vis = predict_function(self.vis, self.model)
        assert numpy.max(numpy.abs(self.vis.vis)) > 0.0
        
        dphasecentre = SkyCoord(ra=+181.0 * u.deg, dec=-58.0 * u.deg, frame='icrs', equinox='J2000')
        flux = [[numpy.power(f/1e8, -0.7)] for f in self.component_frequency]
        self.sc = create_skycomponent(direction=dphasecentre, flux=flux,
                                    frequency=self.component_frequency,
                                    polarisation_frame=PolarisationFrame('stokesI'))

        
    def test_insert_skycomponent_FFT(self):
        
        self.model.data *= 0.0
        self.sc = create_skycomponent(direction=self.phasecentre, flux=self.sc.flux,
                                    frequency=self.component_frequency,
                                    polarisation_frame=PolarisationFrame('stokesI'))

        insert_skycomponent(self.model, self.sc)
        npixel = self.model.shape[3]
        # WCS is 1-relative
        rpix = numpy.round(self.model.wcs.wcs.crpix).astype('int') - 1
        assert rpix[0] == npixel // 2
        assert rpix[1] == npixel // 2
        # The phase centre is at rpix[0], rpix[1] in 0-relative pixels
        assert self.model.data[2, 0, rpix[1], rpix[0]] == 1.0
        # If we predict the visibility, then the imaginary part must be zero. This is determined entirely
        # by shift_vis_to_image in libs.imaging.base
        self.vis.data['vis'][...] = 0.0
        self.vis = predict_function(self.vis, self.model)
        # The actual phase centre of a numpy FFT is at nx //2, nx //2 (0 rel).
        assert numpy.max(numpy.abs(self.vis.vis.imag)) <1e-3

    def test_insert_skycomponent_dft(self):
        self.sc = create_skycomponent(direction=self.phasecentre, flux=self.sc.flux,
                                    frequency=self.component_frequency,
                                    polarisation_frame=PolarisationFrame('stokesI'))

        self.vis.data['vis'][...] = 0.0
        self.vis = predict_skycomponent_visibility(self.vis, self.sc)
        im, sumwt = invert_function(self.vis, self.model)
        export_image_to_fits(im, '%s/test_skycomponent_dft.fits' % self.dir)
        assert numpy.max(numpy.abs(self.vis.vis.imag)) < 1e-3
    
    def test_insert_skycomponent_nearest(self):
        self.model.data *= 0.0
        insert_skycomponent(self.model, self.sc, insert_method='Nearest')
        # These test a regression but are not known a priori to be correct
        self.assertAlmostEqual(self.model.data[2, 0, 151, 122], 1.0, 7)
        self.assertAlmostEqual(self.model.data[2, 0, 152, 122], 0.0, 7)
    
    def test_insert_skycomponent_sinc(self):
        self.model.data *= 0.0
        insert_skycomponent(self.model, self.sc, insert_method='Sinc')
        # These test a regression but are not known a priori to be correct
        self.assertAlmostEqual(self.model.data[2, 0, 151, 122], 0.87684398703184396, 7)
        self.assertAlmostEqual(self.model.data[2, 0, 152, 122], 0.2469311811046056, 7)
    
    def test_insert_skycomponent_sinc_bandwidth(self):
        self.model.data *= 0.0
        insert_skycomponent(self.model, self.sc, insert_method='Sinc', bandwidth=0.5)
        # These test a regression but are not known a priori to be correct
        self.assertAlmostEqual(self.model.data[2, 0, 151, 122], 0.25133066186805758, 7)
        self.assertAlmostEqual(self.model.data[2, 0, 152, 122], 0.19685222464041874, 7)
    
    def test_insert_skycomponent_lanczos(self):
        self.model.data *= 0.0
        insert_skycomponent(self.model, self.sc, insert_method='Lanczos')
        # These test a regression but are not known a priori to be correct
        self.assertAlmostEqual(self.model.data[2, 0, 151, 122], 0.87781267543090036, 7)
        self.assertAlmostEqual(self.model.data[2, 0, 152, 122], 0.23817562762032077, 7)
    
    def test_insert_skycomponent_lanczos_bandwidth(self):
        self.model.data *= 0.0
        insert_skycomponent(self.model, self.sc, insert_method='Lanczos', bandwidth=0.5)
        # These test a regression but are not known a priori to be correct
        self.assertAlmostEqual(self.model.data[2, 0, 151, 122], 0.24031092091707615, 7)
        self.assertAlmostEqual(self.model.data[2, 0, 152, 122], 0.18648989466050975, 7)


if __name__ == '__main__':
    unittest.main()
