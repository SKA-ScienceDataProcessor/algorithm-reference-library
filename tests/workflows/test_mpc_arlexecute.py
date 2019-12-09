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
from workflows.arlexecute.skymodel.skymodel_arlexecute import predict_skymodel_list_arlexecute_workflow, \
    invert_skymodel_list_arlexecute_workflow, crosssubtract_datamodels_skymodel_list_arlexecute_workflow
from workflows.shared.imaging.imaging_shared import sum_predict_results
from wrappers.arlexecute.execution_support.arlexecutebase import ARLExecuteBase
from wrappers.arlexecute.execution_support.dask_init import get_dask_Client
from processing_components.simulation import ingest_unittest_visibility, \
    create_low_test_skymodel_from_gleam
from processing_components.simulation import create_named_configuration
from processing_components.visibility.base import copy_visibility
from processing_components.visibility.coalesce import convert_blockvisibility_to_visibility

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestMPC(unittest.TestCase):
    def setUp(self):
        
        client = get_dask_Client(memory_limit=4 * 1024 * 1024 * 1024, n_workers=4, dashboard_address=None)
        global arlexecute
        arlexecute = ARLExecuteBase(use_dask=True)
        arlexecute.set_client(client)
        
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        self.plot = False
        self.persist = False
    
    def tearDown(self):
        global arlexecute
        arlexecute.close()
        del arlexecute

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
        self.blockvis_list = [arlexecute.execute(ingest_unittest_visibility)(self.low,
                                                                             [self.frequency[freqwin]],
                                                                             [self.channelwidth[freqwin]],
                                                                             self.times,
                                                                             self.vis_pol,
                                                                             self.phasecentre, block=block,
                                                                             zerow=zerow)
                              for freqwin, _ in enumerate(self.frequency)]
        self.blockvis_list = arlexecute.compute(self.blockvis_list, sync=True)
        self.vis_list = [arlexecute.execute(convert_blockvisibility_to_visibility)(bv) for bv in self.blockvis_list]
        self.vis_list = arlexecute.compute(self.vis_list, sync=True)
        
        self.skymodel_list = [arlexecute.execute(create_low_test_skymodel_from_gleam)
                              (npixel=self.npixel, cellsize=self.cellsize, frequency=[self.frequency[f]],
                               phasecentre=self.phasecentre,
                               polarisation_frame=PolarisationFrame("stokesI"),
                               flux_limit=0.6,
                               flux_threshold=1.0,
                               flux_max=5.0) for f, freq in enumerate(self.frequency)]
        
        self.skymodel_list = arlexecute.compute(self.skymodel_list, sync=True)
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
        
        future_vis = arlexecute.scatter(self.vis_list[0])
        future_skymodel = arlexecute.scatter(self.skymodel_list)
        skymodel_vislist = predict_skymodel_list_arlexecute_workflow(future_vis, future_skymodel,
                                                                     context='2d', docal=True)
        skymodel_vislist = arlexecute.compute(skymodel_vislist, sync=True)
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
        
        future_vis = arlexecute.scatter(self.vis_list[0])
        future_skymodel = arlexecute.scatter(self.skymodel_list)
        skymodel_vislist = predict_skymodel_list_arlexecute_workflow(future_vis, future_skymodel,
                                                                    context='2d', docal=True)
        skymodel_vislist = arlexecute.compute(skymodel_vislist, sync=True)
        
        result_skymodel = [SkyModel(components=None, image=self.skymodel_list[-1].image)
                           for v in skymodel_vislist]
        
        self.vis_list = arlexecute.scatter(self.vis_list)
        result_skymodel = invert_skymodel_list_arlexecute_workflow(skymodel_vislist, result_skymodel,
                                                                   context='2d', docal=True)
        results = arlexecute.compute(result_skymodel, sync=True)
        assert numpy.max(numpy.abs(results[0][0].data)) > 0.0
        assert numpy.max(numpy.abs(results[0][1])) > 0.0
        if self.plot:
            import matplotlib.pyplot as plt
            from processing_components.image.operations import show_image
            show_image(results[0][0], title='Dirty image, no cross-subtraction', vmax=0.1, vmin=-0.01)
            plt.show()
    
    def test_crosssubtract_datamodel(self):
        self.actualSetUp(zerow=True)
        
        future_vis = arlexecute.scatter(self.vis_list[0])
        future_skymodel_list = arlexecute.scatter(self.skymodel_list)
        skymodel_vislist = predict_skymodel_list_arlexecute_workflow(future_vis, future_skymodel_list,
                                                                    context='2d', docal=True)
        skymodel_vislist = arlexecute.compute(skymodel_vislist, sync=True)
        vobs = sum_predict_results(skymodel_vislist)
        
        future_vobs = arlexecute.scatter(vobs)
        skymodel_vislist = crosssubtract_datamodels_skymodel_list_arlexecute_workflow(future_vobs, skymodel_vislist)
        
        skymodel_vislist = arlexecute.compute(skymodel_vislist, sync=True)
        
        result_skymodel = [SkyModel(components=None, image=self.skymodel_list[-1].image)
                           for v in skymodel_vislist]
        
        self.vis_list = arlexecute.scatter(self.vis_list)
        result_skymodel = invert_skymodel_list_arlexecute_workflow(skymodel_vislist, result_skymodel,
                                                                   context='2d', docal=True)
        results = arlexecute.compute(result_skymodel, sync=True)
        assert numpy.max(numpy.abs(results[0][0].data)) > 0.0
        assert numpy.max(numpy.abs(results[0][1])) > 0.0
        if self.plot:
            import matplotlib.pyplot as plt
            from processing_components.image.operations import show_image
            show_image(results[0][0], title='Dirty image after cross-subtraction', vmax=0.1, vmin=-0.01)
            plt.show()


if __name__ == '__main__':
    unittest.main()
