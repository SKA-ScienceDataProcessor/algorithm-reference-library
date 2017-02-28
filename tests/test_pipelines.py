"""Unit tests for pipelines

realtimcornwell@gmail.com
"""

import unittest

from arl.fourier_transforms.ftprocessor import *
from arl.pipelines.functions import *
from arl.skymodel.operations import create_skycomponent
from arl.util.run_unittests import run_unittests
from arl.util.testing_support import create_named_configuration, create_test_image, create_blockvisibility_iterator


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
        
        self.comp = create_skycomponent(flux=self.flux, frequency=frequency, direction=self.compabsdirection)
        self.image = create_test_image(frequency=frequency, npol=1, phasecentre=self.phasecentre, cellsize=0.001)
        
        self.vis = create_blockvisibility_iterator(lowcore, times=times, freq=frequency, phasecentre=self.phasecentre,
                                                   weight=1, pol_frame=Polarisation_Frame('stokesIQUV'),
                                                   integration_time=1.0, number_integrations=1, channel_bandwidth=1e6,
                                                   predict=predict_2d, components=self.comp,
                                                   phase_error=0.1, amplitude_error=0.01)
    
    def test_RCAL(self):
        for igt, gt in enumerate(RCAL(vis=self.vis, components=self.comp)):
            print("Chunk %d, gaintable size %.3f (GB)" % (igt, gt.size()))
    
    def test_ICAL(self):
        ical = ICAL(vis=self.visibility, skymodel=self.sm, Gsolinit=300.0)
    
    def test_continuum_imaging(self):
        ci = continuum_imaging(vis=self.visibility, skymodel=self.sm, algorithm='msclean')
    
    def test_spectral_line_imaging(self):
        sli = spectral_line_imaging(vis=self.visibility, skymodel=self.sm, algorithm='msclean')
    
    def test_fast_imaging(self):
        fi = fast_imaging(vis=self.visibility, skymodel=self.sm, Gsolinit=300.0)
    
    def test_EOR(self):
        eor = EOR(vis=self.visibility, skymodel=self.sm, Gsolinit=300.0)


if __name__ == '__main__':
    run_unittests()
