"""Unit tests for pipelines

realtimcornwell@gmail.com
"""

import unittest

from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.fourier_transforms.ftprocessor import *
from arl.image.deconvolution import _msclean
from arl.skymodel.operations import create_skycomponent
from arl.skymodel.operations import create_skymodel_from_image, add_component_to_skymodel
from arl.util.testing_support import create_named_configuration, create_test_image
from arl.visibility.operations import create_visibility
from arl.pipelines.functions import *

class TestPipelines(unittest.TestCase):


    def setUp(self):
        
        
        vlaa = create_named_configuration('VLAA')
        vlaa.data['xyz'] *= 1.0 / 10.0
        times = numpy.arange(-3.0, +3.0, 6.0 / 60.0) * numpy.pi / 12.0
        frequency = numpy.arange(1.0e8, 1.50e8, 2.0e7)
        
        # Define the component and give it some polarisation and spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=+35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=+36.5 * u.deg, frame='icrs', equinox=2000.0)
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)

        self.m31comp = create_skycomponent(flux=self.flux, frequency=frequency, direction=self.compreldirection)
        self.m31image = create_test_image(nchan=3, npol=4)
        cellsize = 180.0 * 0.0001 / numpy.pi
        self.m31image.wcs.wcs.cdelt[0] = -cellsize
        self.m31image.wcs.wcs.cdelt[1] = +cellsize
        self.m31image.wcs.wcs.radesys = 'ICRS'
        self.m31image.wcs.wcs.equinox = 2000.00
        self.m31sm = create_skymodel_from_image(self.m31image)
        self.m31sm = add_component_to_skymodel(self.m31sm, self.m31comp)

        vispred = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=self.phasecentre)
        self.visibility = predict_2d(vispred, self.m31image, wstep=100.0, npixel=256, cellsize=0.0001)
        self.m31image.data *= 0.0
        self.m31sm = create_skymodel_from_image(self.m31image)
        self.m31sm = add_component_to_skymodel(self.m31sm, self.m31comp)

    def test_RCAL(self):
        rcal = RCAL(vis=self.visibility, skymodel=self.m31sm, Gsolinit=300.0)

    def test_ICAL(self):
        ical = ICAL(vis=self.visibility, skymodel=self.m31sm, Gsolinit=300.0)

    def test_continuum_imaging(self):
        ci = continuum_imaging(vis=self.visibility, skymodel=self.m31sm, algorithm='msclean')

    def test_spectral_line_imaging(self):
        sli = spectral_line_imaging(vis=self.visibility, skymodel=self.m31sm, algorithm='msclean')
        

if __name__ == '__main__':
    import sys
    import logging
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    unittest.main()
