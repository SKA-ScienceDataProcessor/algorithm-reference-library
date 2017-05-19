"""Unit tests for pipelines expressed via dask.delayed


"""

import unittest

from dask import delayed
from arl.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility
from arl.image.operations import export_image_to_fits
from arl.pipelines.dask_graphs import create_continuum_imaging_graph, create_predict_graph, \
    create_invert_graph
from arl.pipelines.functions import *
from arl.skycomponent.operations import create_skycomponent
from arl.util.testing_support import create_named_configuration, create_test_image, simulate_gaintable
from arl.fourier_transforms.ftprocessor import invert_timeslice_single
from arl.pipelines.dask_init import kill_dask_Client, get_dask_Client


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
                                          phasecentre=self.phasecentre, weight=1,
                                          polarisation_frame=PolarisationFrame('stokesI'), integration_time=1.0,
                                          channel_bandwidth=self.channel_bandwidth)
        self.predict_graph = create_predict_graph(self.vis, self.image_graph)
        self.vis = self.predict_graph.compute()

    def test_invert_graph(self):
        vis = create_visibility(self.lowcore, times=self.times, frequency=self.frequency,
                                channel_bandwidth=self.channel_bandwidth,
                                phasecentre=self.phasecentre, weight=1,
                                polarisation_frame=PolarisationFrame('stokesI'),
                                integration_time=1.0)
    
        make_graph = create_invert_graph(vis, self.image_graph, dopsf=True, invert_single=invert_timeslice_single,
                                         iterator=vis_timeslice_iter, normalize=True, timeslice='auto', context='')
        psf, sumwt = make_graph.compute()
        assert numpy.max(psf.data) > 0.0
        export_image_to_fits(psf, "%s/test_pipelines-invert-graph-psf.fits" % (self.dir))

    @unittest.skip("Does bad things to jenkins build")
    def test_invert_graph_with_client(self):
        vis = create_visibility(self.lowcore, times=self.times, frequency=self.frequency,
                                channel_bandwidth=self.channel_bandwidth,
                                phasecentre=self.phasecentre, weight=1,
                                polarisation_frame=PolarisationFrame('stokesI'),
                                integration_time=1.0)
    
        make_graph = create_invert_graph(vis, self.image_graph, dopsf=True, invert_single=invert_timeslice_single,
                                         iterator=vis_timeslice_iter, normalize=True, timeslice='auto', context='')
        c=get_dask_Client()
        future = c.compute(make_graph)
        psf, sumwt = future.result()
        assert numpy.max(psf.data) > 0.0
        export_image_to_fits(psf, "%s/test_pipelines-invert-graph-psf.fits" % (self.dir))
        c.shutdown()

    def test_continuum_imaging_graph_directmodel(self):
        continuum_imaging_graph = delayed(create_continuum_imaging_graph)(self.vis,
                                                                          model_graph=self.image_graph.compute(),
                                                                          algorithm='hogbom', niter=1000,
                                                                          fractional_threshold=0.1, threshold=1.0,
                                                                          nmajor=3, gain=0.1)
        comp = continuum_imaging_graph.compute().compute()
        export_image_to_fits(comp, "%s/test_pipelines-continuum-imaging_direct-dask-comp.fits" % (self.dir))
    
    def test_continuum_imaging_graph(self):
        continuum_imaging_graph = create_continuum_imaging_graph(self.vis, model_graph=self.image_graph,
                                                                 algorithm='hogbom',
                                                                 niter=1000, fractional_threshold=0.1,
                                                                 threshold=1.0, nmajor=3, gain=0.1)
        comp = continuum_imaging_graph.compute()
        export_image_to_fits(comp, "%s/test_pipelines-continuum-imaging-dask-comp.fits" % (self.dir))
    
    def test_create_solve_gain_graph(self):
        self.blockvis = create_blockvisibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                          channel_bandwidth=self.channel_bandwidth, weight=1.0,
                                          polarisation_frame=PolarisationFrame('stokesI'))
        self.blockvis = predict_skycomponent_blockvisibility(self.blockvis, self.comp)
        self.modelblockvis = copy_visibility(self.blockvis)
        gt = create_gaintable_from_blockvisibility(self.blockvis)
        gt = simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.0)
        self.blockvis = apply_gaintable(self.blockvis, gt)
        self.modelblockvis = copy_visibility(self.blockvis)
        solve_gain_graph = create_solve_gain_graph(self.blockvis, self.modelblockvis)
        fullgt = solve_gain_graph.compute()
        residual = numpy.max(fullgt.residual)
        assert residual < 3e-8, "Max residual = %s" % (residual)


if __name__ == '__main__':
    unittest.main()
