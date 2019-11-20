""" Unit tests for pipelines expressed via arlexecute
"""

import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.memory_data_models import Image, SkyModel
from data_models.memory_data_models import Skycomponent
from data_models.polarisation import PolarisationFrame
from processing_components.skymodel.operations import expand_skymodel_by_skycomponents
from workflows.serial.skymodel.skymodel_serial import predict_skymodel_list_serial_workflow, \
    invert_skymodel_list_serial_workflow, crosssubtract_datamodels_skymodel_list_serial_workflow
from workflows.shared.imaging.imaging_shared import sum_predict_results
from processing_components.simulation.testing_support import ingest_unittest_visibility, \
    create_low_test_skymodel_from_gleam
from processing_components.simulation.configurations import create_named_configuration
from processing_components.visibility.base import copy_visibility
from processing_components.visibility.coalesce import convert_blockvisibility_to_visibility

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestMPC(unittest.TestCase):
    def setUp(self):
        
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        self.plot = False
        self.persist = False
    
    def actualSetUp(self, freqwin=1, block=True, dopol=False, zerow=False):
        
        self.npixel = 1024
        self.low = create_named_configuration('LOWBD2', rmax=550.0)
        self.freqwin = freqwin
        self.blockvis_list = list()
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
        
        self.phasecentre = SkyCoord(ra=+0.0 * u.deg, dec=-40.0 * u.deg, frame='icrs', equinox='J2000')
        self.blockvis_list = [ingest_unittest_visibility(self.low,
                                                                             [self.frequency[freqwin]],
                                                                             [self.channelwidth[freqwin]],
                                                                             self.times,
                                                                             self.vis_pol,
                                                                             self.phasecentre, block=block,
                                                                             zerow=zerow)
                              for freqwin, _ in enumerate(self.frequency)]
        self.vis_list = [convert_blockvisibility_to_visibility(bv) for bv in self.blockvis_list]
        
        self.skymodel_list = [create_low_test_skymodel_from_gleam
                              (npixel=self.npixel, cellsize=self.cellsize, frequency=[self.frequency[f]],
                               phasecentre=self.phasecentre,
                               polarisation_frame=PolarisationFrame("stokesI"),
                               flux_limit=0.6,
                               flux_threshold=1.0,
                               flux_max=5.0) for f, freq in enumerate(self.frequency)]
        
        assert isinstance(self.skymodel_list[0].image, Image), self.skymodel_list[0].image
        assert isinstance(self.skymodel_list[0].components[0], Skycomponent), self.skymodel_list[0].components[0]
        assert len(self.skymodel_list[0].components) == 35, len(self.skymodel_list[0].components)
        self.skymodel_list = expand_skymodel_by_skycomponents(self.skymodel_list[0])
        assert len(self.skymodel_list) == 36, len(self.skymodel_list)
        assert numpy.max(numpy.abs(self.skymodel_list[-1].image.data)) > 0.0, "Image is empty"
        self.vis_list = [copy_visibility(self.vis_list[0], zero=True) for i, _ in enumerate(self.skymodel_list)]
    
    def test_time_setup(self):
        self.actualSetUp()
    
    def test_predictcal(self):
        
        self.actualSetUp(zerow=True)
        
        skymodel_vislist = predict_skymodel_list_serial_workflow(self.vis_list[0], self.skymodel_list,
                                                                     context='2d', docal=True)
        vobs = sum_predict_results(skymodel_vislist)
        
        if self.plot:
            def plotvis(i, v):
                import matplotlib.pyplot as plt
                uvr = numpy.hypot(v.u, v.v)
                amp = numpy.abs(v.vis[:, 0])
                plt.plot(uvr, amp, '.')
                plt.title(str(i))
                plt.show()
            
            plotvis(0, vobs)
    
    def test_invertcal(self):
        self.actualSetUp(zerow=True)
        
        skymodel_vislist = predict_skymodel_list_serial_workflow(self.vis_list[0], self.skymodel_list,
                                                                     context='2d', docal=True)
        result_skymodel = [SkyModel(components=None, image=self.skymodel_list[-1].image)
                           for v in skymodel_vislist]
        
        results = invert_skymodel_list_serial_workflow(skymodel_vislist, result_skymodel,
                                                                   context='2d', docal=True)
        assert numpy.max(numpy.abs(results[0][0].data)) > 0.0
        assert numpy.max(numpy.abs(results[0][1])) > 0.0
        if self.plot:
            import matplotlib.pyplot as plt
            from processing_components.image.operations import show_image
            show_image(results[0][0], title='Dirty image, no cross-subtraction', vmax=0.1, vmin=-0.01)
            plt.show()
    
    def test_crosssubtract_datamodel(self):
        self.actualSetUp(zerow=True)
        
        skymodel_vislist = predict_skymodel_list_serial_workflow(self.vis_list[0], self.skymodel_list,
                                                                     context='2d', docal=True)
        vobs = sum_predict_results(skymodel_vislist)
        
        skymodel_vislist = crosssubtract_datamodels_skymodel_list_serial_workflow(vobs, skymodel_vislist)
        
        result_skymodel = [SkyModel(components=None, image=self.skymodel_list[-1].image)
                           for v in skymodel_vislist]
        
        results = invert_skymodel_list_serial_workflow(skymodel_vislist, result_skymodel,
                                                                   context='2d', docal=True)
        assert numpy.max(numpy.abs(results[0][0].data)) > 0.0
        assert numpy.max(numpy.abs(results[0][1])) > 0.0
        if self.plot:
            import matplotlib.pyplot as plt
            from processing_components.image.operations import show_image
            show_image(results[0][0], title='Dirty image after cross-subtraction', vmax=0.1, vmin=-0.01)
            plt.show()


if __name__ == '__main__':
    unittest.main()
