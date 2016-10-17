"""Unit tests for Fourier transforms

realtimcornwell@gmail.com
"""
import unittest

import numpy
from numpy.testing import assert_allclose

from astropy.coordinates import SkyCoord
from astropy import units as u

from arl.skymodel_operations import create_skycomponent
from arl.testing_support import create_named_configuration, create_test_image
from arl.image_operations import export_image_to_fits
from arl.skymodel_operations import create_skymodel_from_component, find_skycomponent, fit_skycomponent
from arl.visibility_operations import create_visibility, sum_visibility
from arl.fourier_transforms import predict_visibility, invert_visibility

import logging
log = logging.getLogger( "tests.test_fourier_transforms" )


class TestFourierTransforms(unittest.TestCase):

    def setUp(self):
        
        self.params = {'wstep': 10.0, 'npixel': 512, 'cellsize':0.0002, 'spectral_mode': 'channel'}

        vlaa = create_named_configuration('VLAA')
        vlaa.data['xyz'] *= 1.0 / 30.0
        times = numpy.arange(-3.0, +3.0, 6.0 / 60.0) * numpy.pi / 12.0
        frequency = numpy.arange(1.0e8, 1.50e8, 2.0e7)

        # Define the component and give it some spectral behaviour
        self.model = create_test_image()
        f=numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f,0.8*f,0.6*f])
        self.average = numpy.average(self.flux[:,0])
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre      = SkyCoord(ra=+15.0*u.deg, dec=+35.0*u.deg, frame='icrs', equinox=2000.0)
        self.compabsdirection = SkyCoord(ra=17.0*u.deg,  dec=+36.5*u.deg, frame='icrs', equinox=2000.0)
        # TODO: convert entire mechanism to absolute coordinates
        pcof=self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)
        self.comp = create_skycomponent(flux=self.flux, frequency=frequency, direction=self.compreldirection)
        self.sm = create_skymodel_from_component(self.comp)
        vispred = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=self.phasecentre,
                                   params=self.params)
        self.vismodel = predict_visibility(vispred, self.sm, self.params)
        

    def test_all(self):
        
        # Sum the visibilities in the correct_visibility direction. This is limited by numerical precision
        summedflux, weight = sum_visibility(self.vismodel, self.compreldirection)
        assert_allclose(self.flux, summedflux , rtol=1e-7)

        # Now make a dirty image
        # Check that the flux at the peak is as expected
        self.dirty, self.psf, sumwt = invert_visibility(self.vismodel, self.model, self.params)
        export_image_to_fits(self.dirty, 'test_imaging_dirty.fits')
        log.debug("Max, min in dirty Image = %.6f, %.6f, sum of weights = %f" %
                  (self.dirty.data.max(), self.dirty.data.min(), sumwt))
        log.debug("Max, min in PSF         = %.6f, %.6f, sum of weights = %f" %
                  (self.psf.data.max(), self.psf.data.min(), sumwt))
        # Find the peak, and check that the returned component is correct
        newcomp = find_skycomponent(self.dirty)
        assert_allclose(self.compabsdirection.ra.value,  newcomp.direction.ra.value,  atol=1e-2)
        assert_allclose(self.compabsdirection.dec.value, newcomp.direction.dec.value, atol=1e-2)
        assert_allclose(self.flux, newcomp.flux , rtol=0.05)

        # Check that the returned component is correct
        newcomp = fit_skycomponent(self.dirty, self.compabsdirection)
        # TODO: Track down reason for terrible precision
        assert_allclose(self.compabsdirection.ra.value, newcomp.direction.ra.value, atol=1e-2)
        assert_allclose(self.compabsdirection.dec.value, newcomp.direction.dec.value, atol=1e-2)
        assert_allclose(self.flux, newcomp.flux, rtol=0.05)

if __name__ == '__main__':
    import sys
    import logging
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    unittest.main()
