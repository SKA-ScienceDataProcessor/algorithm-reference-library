"""Unit tests for pipelines

realtimcornwell@gmail.com
"""

import unittest

from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.fourier_transforms.ftprocessor import *
from arl.skymodel.operations import create_skycomponent
from arl.skymodel.operations import create_skymodel_from_image
from arl.util.testing_support import create_named_configuration, create_test_image, run_unittests
from arl.visibility.operations import create_visibility
from arl.pipelines.functions import *

class TestPipelines(unittest.TestCase):

    def setUp(self):
        
        
        lowcore = create_named_configuration('LOWBD2-CORE')
        times = numpy.arange(-3.0, +3.0, 1.0) * numpy.pi / 12.0
        frequency = numpy.linspace(1.0e8, 1.50e8, 3)
        
        # Define the component and give it some polarisation and spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox=2000.0)
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)

        self.m31comp = create_skycomponent(flux=self.flux, frequency=frequency, direction=self.compreldirection)
        self.m31image = create_test_image(frequency=frequency, npol=4, phasecentre=self.phasecentre, cellsize=0.001)
        self.m31sm = create_skymodel_from_image(self.m31image)

        vispred = create_visibility(lowcore, times, frequency, phasecentre=self.phasecentre, weight=1.0)
        self.visibility = predict_2d(vispred, self.m31image, wstep=100.0, npixel=256, cellsize=0.0001)
        self.m31image.data *= 0.0
        self.m31sm = create_skymodel_from_image(self.m31image)

    def test_RCAL(self):
        rcal = RCAL(vis=self.visibility, skymodel=self.m31sm, Gsolinit=300.0)

    def test_ICAL(self):
        ical = ICAL(vis=self.visibility, skymodel=self.m31sm, Gsolinit=300.0)

    def test_continuum_imaging(self):
        ci = continuum_imaging(vis=self.visibility, skymodel=self.m31sm, algorithm='msclean')

    def test_spectral_line_imaging(self):
        sli = spectral_line_imaging(vis=self.visibility, skymodel=self.m31sm, algorithm='msclean')
        

if __name__ == '__main__':
    run_unittests()
