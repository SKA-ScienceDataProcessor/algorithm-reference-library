"""Unit tests for pipelines expressed via dask.delayed


"""
import os
import numpy

import unittest

from dask import delayed

from astropy.coordinates import SkyCoord
import astropy.units as u

from arl.data.polarisation import PolarisationFrame
from arl.data.data_models import BlockVisibility

from arl.fourier_transforms.ftprocessor import invert_timeslice_single
from arl.image.operations import export_image_to_fits
from arl.pipelines.dask_graphs import create_continuum_imaging_graph, create_invert_graph, create_predict_graph, \
    create_ical_graph
from arl.pipelines.dask_init import get_dask_Client
from arl.visibility.operations import create_blockvisibility
from arl.util.testing_support import create_named_configuration, create_test_image


class TestPipelinesDask(unittest.TestCase):
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
        
        self.image_graph = delayed(create_test_image)(frequency=self.frequency, phasecentre=self.phasecentre,
                                                      cellsize=0.001,
                                                      polarisation_frame=PolarisationFrame('stokesI'))
        
        self.vis = [delayed(create_blockvisibility, nout=1)(self.lowcore, times=self.times, frequency=self.frequency,
                                           phasecentre=self.phasecentre, weight=1,
                                           polarisation_frame=PolarisationFrame('stokesI'), integration_time=1.0,
                                           channel_bandwidth=self.channel_bandwidth)]
        self.predict_graph = create_predict_graph(self.vis, self.image_graph)
    
    def test_invert_graph(self):
        make_graph = create_invert_graph(self.vis, self.image_graph, dopsf=True, invert_single=invert_timeslice_single)
        psf, sumwt = make_graph.compute()
        assert numpy.max(psf.data) > 0.0
        export_image_to_fits(psf, "%s/test_pipelines-invert-graph-psf.fits" % (self.dir))
    
    @unittest.skip("Does bad things to jenkins build")
    def test_invert_graph_with_client(self):
        make_graph = create_invert_graph(self.vis, self.image_graph, dopsf=True, invert_single=invert_timeslice_single,
                                         timeslice='auto', context='')
        c = get_dask_Client()
        future = c.compute(make_graph)
        psf, sumwt = future.result()
        assert numpy.max(psf.data) > 0.0
        export_image_to_fits(psf, "%s/test_pipelines-invert-graph-psf.fits" % (self.dir))
        c.shutdown()
    
    def test_continuum_imaging_graph(self):
        continuum_imaging_graph = create_continuum_imaging_graph(self.vis, model_graph=self.image_graph,
                                                                 algorithm='hogbom',
                                                                 niter=1000, fractional_threshold=0.1,
                                                                 threshold=1.0, nmajor=3, gain=0.1)
        comp = continuum_imaging_graph.compute()
        export_image_to_fits(comp[0], "%s/test_pipelines-continuum-imaging-dask-comp.fits" % (self.dir))
        export_image_to_fits(comp[1][0], "%s/test_pipelines-continuum-imaging-dask-residual.fits" % (self.dir))
        export_image_to_fits(comp[2], "%s/test_pipelines-continuum-imaging-dask-restored.fits" % (self.dir))

    def test_ical_graph(self):
        ical_graph = create_ical_graph(self.vis, model_graph=self.image_graph,
                                       algorithm='hogbom',
                                       niter=1000, fractional_threshold=0.1,
                                       threshold=1.0, nmajor=3, first_selfcal=1,
                                       gain=0.1)
        comp = ical_graph.compute()
        export_image_to_fits(comp[0], "%s/test_pipelines-ical-dask-comp.fits" % (self.dir))
        export_image_to_fits(comp[1][0], "%s/test_pipelines-ical-dask-residual.fits" % (self.dir))
        export_image_to_fits(comp[2], "%s/test_pipelines-ical-dask-restored.fits" % (self.dir))

if __name__ == '__main__':
    unittest.main()
