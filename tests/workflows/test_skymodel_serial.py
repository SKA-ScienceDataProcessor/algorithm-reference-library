""" Unit tests for pipelines expressed via arlexecute
"""

import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from workflows.serial.skymodel.skymodel_serial import predict_skymodel_list_serial_workflow

from data_models.memory_data_models import Image
from data_models.memory_data_models import Skycomponent
from data_models.polarisation import PolarisationFrame
from wrappers.serial.simulation.testing_support import ingest_unittest_visibility, \
    create_low_test_skymodel_from_gleam
from processing_components.simulation.configurations import create_named_configuration

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestSkyModel(unittest.TestCase):
    def setUp(self):
        
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
    
    def tearDown(self):
        pass
    
    def actualSetUp(self, freqwin=1, block=False, dopol=False, zerow=False):
        
        self.npixel = 1024
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = freqwin
        self.vis_list = list()
        self.ntimes = 5
        self.cellsize = 0.0005
        # Choose the interval so that the maximum change in w is smallish
        integration_time = numpy.pi * (24 / (12 * 60))
        self.times = numpy.linspace(-integration_time * (self.ntimes // 2), integration_time * (self.ntimes // 2),
                                    self.ntimes)
        
        if freqwin > 1:
            self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
            self.channelwidth = numpy.array(freqwin * [self.frequency[1] - self.frequency[0]])
        else:
            self.frequency = numpy.array([1.0e8])
            self.channelwidth = numpy.array([4e7])
        
        if dopol:
            self.vis_pol = PolarisationFrame('linear')
            self.image_pol = PolarisationFrame('stokesIQUV')
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        else:
            self.vis_pol = PolarisationFrame('stokesI')
            self.image_pol = PolarisationFrame('stokesI')
            f = numpy.array([100.0])
        
        self.phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis_list = [ingest_unittest_visibility(self.low,
                                                    [self.frequency[freqwin]],
                                                    [self.channelwidth[freqwin]],
                                                    self.times,
                                                    self.vis_pol,
                                                    self.phasecentre, block=block,
                                                    zerow=zerow)
                         for freqwin, _ in enumerate(self.frequency)]
    
    def test_time_setup(self):
        self.actualSetUp()
    
    def test_predict(self):
        self.actualSetUp(zerow=True)
        
        self.skymodel_list = [create_low_test_skymodel_from_gleam
                              (npixel=self.npixel, cellsize=self.cellsize, frequency=[self.frequency[f]],
                               phasecentre=self.phasecentre,
                               polarisation_frame=PolarisationFrame("stokesI"),
                               flux_limit=0.3,
                               flux_threshold=1.0,
                               flux_max=5.0) for f, freq in enumerate(self.frequency)]
        
        assert isinstance(self.skymodel_list[0].image, Image), self.skymodel_list[0].image
        assert isinstance(self.skymodel_list[0].components[0], Skycomponent), self.skymodel_list[0].components[0]
        assert len(self.skymodel_list[0].components) == 25, len(self.skymodel_list[0].components)
        assert numpy.max(numpy.abs(self.skymodel_list[0].image.data)) > 0.0, "Image is empty"
        
        skymodel_vislist = predict_skymodel_list_serial_workflow(self.vis_list[0],
                                                                 self.skymodel_list, context='2d')
        assert numpy.max(numpy.abs(skymodel_vislist[0].vis)) > 0.0


    def test_predict_nocomponents(self):
        self.actualSetUp(zerow=True)
        
        self.skymodel_list = [create_low_test_skymodel_from_gleam
                              (npixel=self.npixel, cellsize=self.cellsize, frequency=[self.frequency[f]],
                               phasecentre=self.phasecentre,
                               polarisation_frame=PolarisationFrame("stokesI"),
                               flux_limit=0.3,
                               flux_threshold=1.0,
                               flux_max=5.0) for f, freq in enumerate(self.frequency)]
        
        
        for i, sm in enumerate(self.skymodel_list):
            sm.components = []
        
        assert isinstance(self.skymodel_list[0].image, Image), self.skymodel_list[0].image
        assert numpy.max(numpy.abs(self.skymodel_list[0].image.data)) > 0.0, "Image is empty"
        
        skymodel_vislist = predict_skymodel_list_serial_workflow(self.vis_list[0], self.skymodel_list, context='2d')
        assert numpy.max(numpy.abs(skymodel_vislist[0].vis)) > 0.0
    
    
    def test_predict_noimage(self):
        self.actualSetUp(zerow=True)
        
        self.skymodel_list = [create_low_test_skymodel_from_gleam
                              (npixel=self.npixel, cellsize=self.cellsize, frequency=[self.frequency[f]],
                               phasecentre=self.phasecentre,
                               polarisation_frame=PolarisationFrame("stokesI"),
                               flux_limit=0.3,
                               flux_threshold=1.0,
                               flux_max=5.0) for f, freq in enumerate(self.frequency)]
        
        for i, sm in enumerate(self.skymodel_list):
            sm.image= None

        assert isinstance(self.skymodel_list[0].components[0], Skycomponent), self.skymodel_list[0].components[0]
        assert len(self.skymodel_list[0].components) == 25, len(self.skymodel_list[0].components)
        
        skymodel_vislist = predict_skymodel_list_serial_workflow(self.vis_list[0], self.skymodel_list, context='2d')
        assert numpy.max(numpy.abs(skymodel_vislist[0].vis)) > 0.0


if __name__ == '__main__':
    unittest.main()
