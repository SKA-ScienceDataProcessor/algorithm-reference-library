"""Unit tests for pipelines expressed via dask.delayed


"""

import unittest

from dask import delayed

from arl.fourier_transforms.ftprocessor import *
from arl.image.operations import export_image_to_fits
from arl.pipelines.dask_graphs import create_continuum_imaging_graph, create_predict_graph, \
    create_solve_gain_graph
from arl.pipelines.functions import *
from arl.skycomponent.operations import create_skycomponent
from arl.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility
from arl.util.testing_support import create_named_configuration, create_test_image, simulate_gaintable


class TestPipelines_dask(unittest.TestCase):
    def setUp(self):
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = numpy.linspace(-3, +3, 13) * (numpy.pi / 12.0)
        
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e7])
        
        # Define the component and give it some polarisation and spectral behaviour
        f = numpy.array([100.0])
        self.flux = numpy.array([f])
        
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox=2000.0)
        
        self.comp = create_skycomponent(flux=self.flux, frequency=self.frequency, direction=self.compabsdirection,
                                        polarisation_frame=PolarisationFrame('stokesI'))
        self.image_graph = delayed(create_test_image)(frequency=self.frequency, phasecentre=self.phasecentre,
                                                      cellsize=0.001,
                                                      polarisation_frame=PolarisationFrame('stokesI'))
        
        self.vis = create_visibility(self.lowcore, times=self.times, frequency=self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
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
        
    def test_create_solve_gain_graph(self):
        self.blockvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                               phasecentre=self.phasecentre,
                                               weight=1.0, polarisation_frame=PolarisationFrame('stokesI'))
        self.blockvis = predict_skycomponent_blockvisibility(self.blockvis, self.comp)
        gt = create_gaintable_from_blockvisibility(self.blockvis)
        gt = simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.0)
        self.blockvis = apply_gaintable(self.blockvis, gt)
        solve_gain_graph = create_solve_gain_graph(self.blockvis, self.comp)
        fullgt = solve_gain_graph.compute()
        residual = numpy.max(fullgt.residual)
        assert residual < 3e-8, "Max residual = %s" % (residual)



if __name__ == '__main__':
    unittest.main()
