import unittest

import numpy
from numpy.testing import assert_allclose

from astropy.coordinates import SkyCoord
from astropy import units as u

from functions.skycomponent import SkyComponent
from functions.configuration import named_configuration, configuration_filter
from functions.image import image_to_fits, fitcomponent, findflux
from functions.skymodel import skymodel_from_component
from functions.visibility import create_visibility, visibility_sum
from functions.imaging import predict, invert


class TestImaging(unittest.TestCase):

    def setUp(self):
        self.kwargs = {'wstep': 10.0, 'npixel': 512, 'cellsize':0.0002}

        vlaa = configuration_filter(named_configuration('VLAA'), **self.kwargs)
        vlaa.data['xyz'] *= 1.0 / 30.0
        times = numpy.arange(-3.0, +3.0, 6.0 / 60.0) * numpy.pi / 12.0
        frequency = numpy.arange(1.0e8, 1.50e8, 2.0e7)

        # Define the component and give it some spectral behaviour
        f=numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f,0.8*f,0.6*f])
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre      = SkyCoord(ra=+15.0*u.deg, dec=+35.0*u.deg, frame='icrs', equinox=2000.0)
        self.compabsdirection = SkyCoord(ra=17.0*u.deg,  dec=+36.5*u.deg, frame='icrs', equinox=2000.0)
        # TODO: convert entire mechanism to absolute coordinates
        pcof=self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)

        self.m31comp = SkyComponent(self.compreldirection, self.flux, frequency)
        self.m31sm = skymodel_from_component(self.m31comp)

        vtpred = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=self.phasecentre,
                                   meta= ** self.kwargs
        self.vtmodel = predict(vtpred, self.m31sm, **self.kwargs)


    def test_visibilitysum(self):
        # Sum the visibilities in the correct direction. This is limited by numerical precision
        summedflux, weight = visibility_sum(self.vtmodel, self.compreldirection)
        assert_allclose(self.flux, summedflux , rtol=1e-7)


    def test_findflux(self):
        # Now make a dirty image
        self.dirty, self.psf, sumwt = invert(self.vtmodel, **self.kwargs)
        image_to_fits(self.dirty, 'test_imaging_dirty.fits')
        print("Max, min in dirty Image = %.6f, %.6f, sum of weights = %f" %
              (self.dirty.data.max(), self.dirty.data.min(), sumwt))
        print("Max, min in PSF         = %.6f, %.6f, sum of weights = %f" %
              (self.psf.data.max(), self.psf.data.min(), sumwt))
        # Find the flux at the location we put it at
        newcomp = findflux(self.dirty, self.compabsdirection)
        # TODO: Track down reason for terrible precision
        assert_allclose(self.flux, newcomp.flux, rtol=0.05)


    def test_fitcomponent(self):
        # Now make a dirty image
        self.dirty, self.psf, sumwt = invert(self.vtmodel, **self.kwargs)
        image_to_fits(self.dirty, 'test_imaging_dirty.fits')
        print("Max, min in dirty Image = %.6f, %.6f, sum of weights = %f" %
              (self.dirty.data.max(), self.dirty.data.min(), sumwt))
        print("Max, min in PSF         = %.6f, %.6f, sum of weights = %f" %
              (self.psf.data.max(), self.psf.data.min(), sumwt))
        # Find the peak
        newcomp = fitcomponent(self.dirty, **self.kwargs)
        # TODO: Track down reason for terrible precision
        assert_allclose(self.flux, newcomp.flux , rtol=0.05)
        # Check that the returned direction is correct
        assert_allclose(self.compabsdirection.ra.value,  newcomp.direction.ra.value,  atol=1e-2)
        assert_allclose(self.compabsdirection.dec.value, newcomp.direction.dec.value, atol=1e-2)

if __name__ == '__main__':
    unittest.main()
