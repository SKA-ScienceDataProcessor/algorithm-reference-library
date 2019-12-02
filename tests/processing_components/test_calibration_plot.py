""" Unit tests for calibration solution


"""
import logging
import unittest

import astropy.units as u
import matplotlib.pyplot as plt
import numpy
from astropy.coordinates import SkyCoord

from data_models.memory_data_models import Skycomponent
from data_models.polarisation import PolarisationFrame
from processing_components.calibration.operations import create_gaintable_from_blockvisibility, \
    gaintable_summary, gaintable_plot
from processing_components.imaging.base import predict_skycomponent_visibility
from processing_components.simulation.configurations import create_named_configuration
from processing_components.simulation.testing_support import simulate_gaintable
from processing_components.visibility.base import create_blockvisibility

log = logging.getLogger(__name__)


class TestCalibrationPlot(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(180555)
    
    def actualSetup(self, sky_pol_frame='stokesIQUV', data_pol_frame='linear', f=None, vnchan=3, ntimes=30):
        self.lowcore = create_named_configuration('LOWBD2', rmax=50.0)
        self.times = (numpy.pi / 43200.0) * numpy.linspace(0.0, 30.0, ntimes)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, vnchan)
        self.channel_bandwidth = numpy.array(vnchan * [self.frequency[1] - self.frequency[0]])
        
        if f is None:
            f = [100.0, 50.0, -10.0, 40.0]
        
        if sky_pol_frame == 'stokesI':
            f = [100.0]
        
        self.flux = numpy.outer(numpy.array([numpy.power(freq / 1e8, -0.7) for freq in self.frequency]), f)
        
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.comp = Skycomponent(direction=self.compabsdirection, frequency=self.frequency, flux=self.flux,
                                 polarisation_frame=PolarisationFrame(sky_pol_frame))
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                          channel_bandwidth=self.channel_bandwidth, weight=1.0,
                                          polarisation_frame=PolarisationFrame(data_pol_frame))
        self.vis = predict_skycomponent_visibility(self.vis, self.comp)
    
    def test_plot_gaintable_scalar(self):
        self.actualSetup('stokesI', 'stokesI', f=[100.0])
        gt = create_gaintable_from_blockvisibility(self.vis)
        log.info("Created gain table: %s" % (gaintable_summary(gt)))
        gt = simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.1)
        plt.clf()
        fig, ax = plt.subplots(1, 1)
        gaintable_plot(gt, ax, value='amp')
        plt.show(block=False)
        fig, ax = plt.subplots(1, 1)
        gaintable_plot(gt, ax, value='phase')
        plt.show(block=False)


if __name__ == '__main__':
    unittest.main()
