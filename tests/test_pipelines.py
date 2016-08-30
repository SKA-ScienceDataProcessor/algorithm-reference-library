"""Unit tests for pipelines

realtimcornwell@gmail.com
"""


import unittest

import numpy
from numpy.testing import assert_allclose

from arl.skymodel_operations import SkyComponent, create_skycomponent
from arl.test_support import create_named_configuration
from arl.image_operations import add_image, create_image_from_array, import_image_from_fits
from arl.skymodel_operations import create_skymodel_from_image, create_skymodel_from_component, \
    add_component_to_skymodel
from arl.visibility_operations import Visibility, create_visibility, create_gaintable_from_array
from arl.fourier_transforms import predict_visibility

from astropy.coordinates import SkyCoord
from astropy import units as u

from arl.pipelines import *


class TestDataPipelines(unittest.TestCase):


    def setUp(self):
        self.kwargs = {'wstep': 10.0, 'npixel': 512, 'cellsize': 0.0002}
        
        vlaa = create_named_configuration('VLAA')
        vlaa.data['xyz'] *= 1.0 / 30.0
        times = numpy.arange(-3.0, +3.0, 6.0 / 60.0) * numpy.pi / 12.0
        frequency = numpy.arange(1.0e8, 1.50e8, 2.0e7)
        
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=+35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=+36.5 * u.deg, frame='icrs', equinox=2000.0)
        # TODO: convert entire mechanism to absolute coordinates
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)
        self.m31comp = create_skycomponent(flux=self.flux, frequency=frequency, direction=self.compreldirection)
        self.m31sm = create_skymodel_from_component(self.m31comp)
        
        vtpred = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=self.phasecentre,
                                   **self.kwargs)
        self.visibility = predict_visibility(vtpred, self.m31sm, **self.kwargs)

    def test_RCAL(self):
        parameters = {'RCAL': {'visibility': self.visibility, 'skymodel': None}}
        rcal = RCAL(parameters)


    def test_ICAL(self):
        parameters = {'ICAL': {'visibility': self.visibility, 'skymodel': self.m31sm}}
        ical = ICAL(parameters)


    def test_continuum_imaging(self):
        parameters = {'continuum_imaging': {'visibility': self.visibility, 'skymodel': self.m31sm}}
        ci = continuum_imaging(parameters)


    def test_spectral_line_imaging(self):
        parameters = {'spectral_line_imaging': {'visibility': self.visibility, 'skymodel': self.m31sm}}
        sli = spectral_line_imaging(parameters)


if __name__ == '__main__':
    unittest.main()
