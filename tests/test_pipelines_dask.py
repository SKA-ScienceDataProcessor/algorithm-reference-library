"""Unit tests for pipelines expressed via dask.delayed


"""

import unittest

from dask import delayed

from arl.fourier_transforms.ftprocessor import *
from arl.image.operations import export_image_to_fits
from arl.pipelines.dask_graphs import create_continuum_imaging_graph, create_predict_graph
from arl.pipelines.functions import *
from arl.skycomponent.operations import create_skycomponent
from arl.util.testing_support import create_named_configuration, create_test_image


class TestPipelines_dask(unittest.TestCase):
    def setUp(self):
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        lowcore = create_named_configuration('LOWBD2-CORE')
        times = numpy.linspace(-3, +3, 13) * (numpy.pi / 12.0)
        
        frequency = numpy.array([1e8])
        channel_bandwidth = numpy.array([1e7])
        
        # Define the component and give it some polarisation and spectral behaviour
        f = numpy.array([100.0])
        self.flux = numpy.array([f])
        
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox=2000.0)
        
        self.comp = create_skycomponent(flux=self.flux, frequency=frequency, direction=self.compabsdirection,
                                        polarisation_frame=PolarisationFrame('stokesI'))
        self.image_graph = delayed(create_test_image)(frequency=frequency, phasecentre=self.phasecentre, cellsize=0.001,
                                       polarisation_frame=PolarisationFrame('stokesI'))
        
        self.vis = create_visibility(lowcore, times=times, frequency=frequency,
                                     channel_bandwidth=channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1,
                                     polarisation_frame=PolarisationFrame('stokesI'),
                                     integration_time=1.0)
        
        self.predict_graph = create_predict_graph(self.vis, self.image_graph)
        self.vis = self.predict_graph.compute()
    
    def test_continuum_imaging_graph(self):
        self.model_graph = delayed(create_empty_image_like)(self.image_graph.compute())
        continuum_imaging_graph = create_continuum_imaging_graph(self.vis, model_graph=self.model_graph,
                                                                 algorithm='hogbom',
                                                                 niter=1000, fractional_threshold=0.1,
                                                                 threshold=1.0, nmajor=3, gain=0.1)
        comp = continuum_imaging_graph.compute()
        export_image_to_fits(comp, "%s/test_pipelines-continuum-imaging-dask-comp.fits" % (self.dir))


if __name__ == '__main__':
    unittest.main()
