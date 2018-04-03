""" Unit tests for pipelines expressed via dask.delayed


"""

import logging
import os
import sys
import unittest

import dask
import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from dask import delayed

from arl.data.polarisation import PolarisationFrame
from arl.graphs.delayed import create_invert_graph, create_deconvolve_graph, create_residual_graph, \
    create_restore_graph
from arl.image.operations import export_image_to_fits, smooth_image
from arl.imaging import predict_skycomponent_visibility
from arl.skycomponent.operations import insert_skycomponent
from arl.util.testing_support import create_named_configuration, ingest_unittest_visibility, create_unittest_model, \
    create_unittest_components, insert_unittest_errors

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImagingDeconvolveDelayed(unittest.TestCase):
    
    def actualSetUp(self, add_errors=False, freqwin=7, block=False, dospectral=True, dopol=False):
        
        dask.set_options(get=dask.get)

        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        self.params = {'npixel': 512,
                       'nchan': 1,
                       'reffrequency': 1e8,
                       'facets': 1,
                       'padding': 2,
                       'oversampling': 2,
                       'kernel': '2d',
                       'wstep': 4.0,
                       'vis_slices': 1,
                       'wstack': None,
                       'timeslice': 'auto'}
        
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = freqwin
        self.vis_graph_list = list()
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        
        if freqwin > 1:
            self.channelwidth = numpy.array(freqwin * [self.frequency[1] - self.frequency[0]])
        else:
            self.channelwidth = numpy.array([1e6])
        
        if dopol:
            self.vis_pol = PolarisationFrame('linear')
            self.image_pol = PolarisationFrame('stokesIQUV')
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        else:
            self.vis_pol = PolarisationFrame('stokesI')
            self.image_pol = PolarisationFrame('stokesI')
            f = numpy.array([100.0])
        
        if dospectral:
            flux = numpy.array([f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency])
        else:
            flux = numpy.array([f])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis_graph_list = [delayed(ingest_unittest_visibility)(self.low,
                                                                   [self.frequency[freqwin]],
                                                                   [self.channelwidth[freqwin]],
                                                                   self.times,
                                                                   self.vis_pol,
                                                                   self.phasecentre, block=block)
                               for freqwin, _ in enumerate(self.frequency)]
        
        self.model_graph = [delayed(create_unittest_model, nout=freqwin)(self.vis_graph_list[freqwin],
                                                                         self.image_pol,
                                                                         npixel=self.params['npixel'])
                            for freqwin, _ in enumerate(self.frequency)]
        
        self.components_graph = [delayed(create_unittest_components)(self.model_graph[freqwin],
                                                                     flux[freqwin, :][numpy.newaxis, :])
                                 for freqwin, _ in enumerate(self.frequency)]
        
        self.model_graph = [delayed(insert_skycomponent, nout=1)(self.model_graph[freqwin],
                                                                 self.components_graph[freqwin])
                            for freqwin, _ in enumerate(self.frequency)]
        
        self.vis_graph_list = [delayed(predict_skycomponent_visibility)(self.vis_graph_list[freqwin],
                                                                        self.components_graph[freqwin])
                               for freqwin, _ in enumerate(self.frequency)]
        
        # Calculate the model convolved with a Gaussian.
        model = self.model_graph[0].compute()
        self.cmodel = smooth_image(model)
        export_image_to_fits(model, '%s/test_imaging_delayed_model.fits' % self.dir)
        export_image_to_fits(self.cmodel, '%s/test_imaging_delayed_cmodel.fits' % self.dir)
        
        if add_errors and block:
            self.vis_graph_list = [delayed(insert_unittest_errors)(self.vis_graph_list[i])
                                   for i, _ in enumerate(self.frequency)]
    
    def test_deconvolve_spectral(self):
        self.actualSetUp(add_errors=True)
        dirty_graph = create_invert_graph(self.vis_graph_list, self.model_graph,
                                          context='wstack', vis_slices=51,
                                          dopsf=False, normalize=True)
        psf_graph = create_invert_graph(self.vis_graph_list, self.model_graph,
                                        context='wstack', vis_slices=51,
                                        dopsf=True, normalize=True)
        dec_graph = create_deconvolve_graph(dirty_graph, psf_graph, self.model_graph, niter=1000,
                                            fractional_threshold=0.1, scales=[0, 3, 10],
                                            threshold=0.1, nmajor=0, gain=0.7)
        deconvolved = dec_graph.compute()
        export_image_to_fits(deconvolved[0], '%s/test_imaging_delayed_deconvolve_spectral.fits' % self.dir)
    
    def test_deconvolve_and_restore_cube_mmclean(self):
        self.actualSetUp(add_errors=True)
        dirty_graph = create_invert_graph(self.vis_graph_list, self.model_graph, context='wstack', vis_slices=51,
                                          dopsf=False, normalize=True)
        psf_graph = create_invert_graph(self.vis_graph_list, self.model_graph, context='wstack', vis_slices=51,
                                        dopsf=True, normalize=True)
        dec_graph = create_deconvolve_graph(dirty_graph, psf_graph, self.model_graph, niter=1000,
                                            fractional_threshold=0.1, scales=[0, 3, 10],
                                            algorithm='mmclean', nmoments=3, nchan=self.freqwin,
                                            threshold=0.1, nmajor=0, gain=0.7)
        residual_graph = create_residual_graph(self.vis_graph_list, model_graph=dec_graph,
                                               context='wstack', vis_slices=51)
        rest_graph = create_restore_graph(model_graph=dec_graph, psf_graph=psf_graph, residual_graph=residual_graph,
                                          empty=self.model_graph)
        restored = rest_graph[0].compute()
        export_image_to_fits(restored, '%s/test_imaging_delayed_mmclean_restored.fits' % self.dir)
    
    def test_deconvolve_and_restore_cube_mmclean_deconvolve_facets(self):
        self.actualSetUp(add_errors=True)
        dirty_graph = create_invert_graph(self.vis_graph_list, self.model_graph,
                                          context='wstack', vis_slices=51,
                                          dopsf=False, normalize=True)
        psf_graph = create_invert_graph(self.vis_graph_list, self.model_graph,
                                        context='wstack', vis_slices=51,
                                        dopsf=True, normalize=True)
        dec_graph = create_deconvolve_graph(dirty_graph, psf_graph, self.model_graph, niter=1000,
                                            fractional_threshold=0.1, scales=[0, 3, 10],
                                            algorithm='mmclean', nmoments=3, nchan=self.freqwin,
                                            threshold=0.1, nmajor=0, gain=0.7, deconvolve_facets=4, deconvolve_overlap=16)
        residual_graph = create_residual_graph(self.vis_graph_list, model_graph=dec_graph,
                                               context='wstack', vis_slices=51)
        rest_graph = create_restore_graph(model_graph=dec_graph, psf_graph=psf_graph, residual_graph=residual_graph,
                                          empty=self.model_graph)
        restored = rest_graph[0].compute()
        export_image_to_fits(restored, '%s/test_imaging_delayed_overlap_mmclean_restored.fits' % self.dir)


if __name__ == '__main__':
    unittest.main()
