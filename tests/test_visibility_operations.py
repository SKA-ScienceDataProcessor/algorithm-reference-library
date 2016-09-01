"""Unit tests for visibility operations

realtimcornwell@gmail.com
"""

import unittest

import numpy
from numpy.testing import assert_allclose

from astropy.coordinates import SkyCoord
from astropy import units as u

from arl.skymodel_operations import create_skycomponent
from arl.test_support import create_named_configuration, filter_configuration
from arl.image_operations import export_image_to_fits
from arl.skymodel_operations import create_skymodel_from_component, find_skycomponent, fit_skycomponent
from arl.visibility_operations import create_visibility, sum_visibility
from arl.fourier_transforms import predict_visibility, invert_visibility


class TestVisibilityOperations(unittest.TestCase):

    def setUp(self):
        self.parameters = {'wstep': 10.0, 'npixel': 512, 'cellsize':0.0002}

        vlaa = filter_configuration(create_named_configuration('VLAA'), self.parameters)
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
        self.m31comp = create_skycomponent(flux=self.flux, frequency=frequency, direction=self.compreldirection)
        self.m31sm = create_skymodel_from_component(self.m31comp)

        vtpred = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=self.phasecentre,
                                   parameters=self.parameters)
        self.vtmodel = predict_visibility(vtpred, self.m31sm, self.parameters)


    def test_visibilitysum(self):
        # Sum the visibilities in the correct_visibility direction. This is limited by numerical precision
        summedflux, weight = sum_visibility(self.vtmodel, self.compreldirection)
        assert_allclose(self.flux, summedflux , rtol=1e-7)

if __name__ == '__main__':
    unittest.main()
