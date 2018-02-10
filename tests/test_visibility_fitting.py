""" Unit tests for visibility operations


"""

import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from arl.data.data_models import Skycomponent
from arl.data.polarisation import PolarisationFrame
from arl.imaging import predict_skycomponent_visibility
from arl.util.testing_support import create_named_configuration
from arl.visibility.base import create_visibility
from arl.visibility.visibility_fitting import fit_visibility


class TestVisibilityFitting(unittest.TestCase):
    def setUp(self):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        self.times = [0.0]
        self.frequency = numpy.linspace(1.0e8, 1.1e8, 1)
        self.channel_bandwidth = numpy.array([1e7])
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        
        f = numpy.array([100.0])
        self.flux = numpy.array([f])
        
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)
        print(self.compreldirection)
        self.comp = Skycomponent(direction=self.compreldirection, frequency=self.frequency, flux=self.flux,
                                 polarisation_frame=PolarisationFrame("stokesI"))
    
    def test_fit_visibility(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("stokesI"))
        self.vismodel = predict_skycomponent_visibility(self.vis, self.comp)
        initial_comp = Skycomponent(direction=self.compreldirection, frequency=self.frequency, flux=1.1 * self.flux,
                                    polarisation_frame=PolarisationFrame("stokesI"))
        
        # Sum the visibilities in the correct_visibility direction. This is limited by numerical precision
        res = fit_visibility(self.vismodel, initial_comp, niter=10)
        print(res)
