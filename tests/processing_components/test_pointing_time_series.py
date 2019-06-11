""" Unit tests for pointing

"""

import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.parameters import arl_path
from data_models.polarisation import PolarisationFrame

from processing_components.skycomponent.operations import create_skycomponent
from processing_components.calibration.pointing import create_pointingtable_from_blockvisibility
from processing_components.imaging.primary_beams import create_vp
from processing_components.simulation.configurations import create_named_configuration
from processing_components.simulation.pointing import create_gaintable_from_pointingtable
from processing_components.simulation.testing_support import create_test_image, simulate_pointingtable, \
    create_pointingtable_from_timeseries
from processing_components.simulation.testing_support import create_test_skycomponents_from_s3
from processing_components.visibility.base import create_blockvisibility
from processing_library.image.operations import create_image

log = logging.getLogger(__name__)


class TestPointing(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        
        self.doplot = True
        
        self.midcore = create_named_configuration('MID', rmax=300.0)
        self.nants = len(self.midcore.names)
        self.dir = arl_path('test_results')
        self.ntimes = 300
        self.times = numpy.linspace(-12.0, 12.0, self.ntimes) * numpy.pi / (12.0)
        
        self.frequency = numpy.array([1e9])
        self.channel_bandwidth = numpy.array([1e7])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_blockvisibility(self.midcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre, weight=1.0,
                                          polarisation_frame=PolarisationFrame('stokesI'))
        self.vis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_image(npixel=512, cellsize=0.00015, polarisation_frame=PolarisationFrame("stokesI"),
                                  frequency=self.frequency, channel_bandwidth=self.channel_bandwidth,
                                  phasecentre=self.phasecentre)
    
    def test_create_gaintable_from_time_series(self):
        s3_components = create_test_skycomponents_from_s3(flux_limit=5.0,
                                                          phasecentre=self.phasecentre,
                                                          frequency=self.frequency,
                                                          polarisation_frame=PolarisationFrame('stokesI'),
                                                          radius=0.2)
        
        pt = create_pointingtable_from_blockvisibility(self.vis)
        pt = create_pointingtable_from_timeseries(pt, scaling=1.0)
        vp = create_vp(self.model, 'MID')
        gt = create_gaintable_from_pointingtable(self.vis, s3_components, pt, vp)
        assert gt[0].gain.shape == (self.ntimes, self.nants, 1, 1, 1), gt[0].gain.shape


if __name__ == '__main__':
    unittest.main()
