import unittest

import numpy
from numpy.testing import assert_allclose

from functions.skycomponent import SkyComponent
from functions.configuration import named_configuration, configuration_filter
from functions.image import image_add, image_from_array, image_from_fits, image_to_fits, image_replicate
from functions.skymodel import skymodel_from_image, skymodel_add_component, skymodel_from_component
from functions.visibility import Visibility, simulate, visibilitysum
from functions.imaging import predict, invert, fitcomponent, findflux

from astropy.coordinates import SkyCoord


class TestImaging(unittest.TestCase):

    def setUp(self):
        self.kwargs = {'wstep': 10.0, 'cellsize': 0.00025, 'npixel': 256}

        vlaa = configuration_filter(named_configuration('VLAA'), **self.kwargs)
        vlaa.data['xyz'] *= 1.0 / 30.0
        times = numpy.arange(-3.0, +3.0, 6.0 / 60.0) * numpy.pi / 12.0
        frequency = numpy.arange(1.0e8, 1.50e8, 2.0e7)

        # Define the component and give it some spectral behaviour
        f=numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f,0.8*f,0.6*f])
        self.compdirection   = SkyCoord(ra="+00d00m00.0s", dec="+00d00m0.00s")
        self.m31comp = SkyComponent(self.compdirection, self.flux, frequency)
        self.m31sm = skymodel_from_component(self.m31comp)

        self.phasecentre = SkyCoord(ra="+00d00m00.0s",  dec="+35d00m0.00s")
        vtpred = simulate(vlaa, times, frequency, weight=1.0, phasecentre=self.phasecentre, **self.kwargs)
        self.vtmodel = predict(vtpred, self.m31sm, **self.kwargs)


    def test_visibilitysum(self):
        # Sum the visibilities in the correct direction. This is limited by numerical precision
        summedflux, weight = visibilitysum(self.vtmodel, self.compdirection)
        assert_allclose(self.flux, summedflux , rtol=1e-7)


    def test_findflux(self):
        # Now make a dirty image
        self.dirty, self.psf, sumwt = invert(self.vtmodel, **self.kwargs)
        print("Max, min in dirty Image = %.6f, %.6f, sum of weights = %f" %
              (self.dirty.data.max(), self.dirty.data.min(), sumwt))
        print("Max, min in PSF         = %.6f, %.6f, sum of weights = %f" %
              (self.psf.data.max(), self.psf.data.min(), sumwt))
        # Find the peak
        sc = findflux(self.dirty, self.vtmodel.phasecentre)
        assert_allclose(self.flux, sc.flux, rtol=1e-5)


    def test_fitcomponent(self):
        # Now make a dirty image
        self.dirty, self.psf, sumwt = invert(self.vtmodel, **self.kwargs)
        print("Max, min in dirty Image = %.6f, %.6f, sum of weights = %f" %
              (self.dirty.data.max(), self.dirty.data.min(), sumwt))
        print("Max, min in PSF         = %.6f, %.6f, sum of weights = %f" %
              (self.psf.data.max(), self.psf.data.min(), sumwt))
        # Find the flux at the location we put it at
        newcomp = fitcomponent(self.dirty, **self.kwargs)
        print(newcomp.frequency)
        assert_allclose(self.flux, newcomp.flux , rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
