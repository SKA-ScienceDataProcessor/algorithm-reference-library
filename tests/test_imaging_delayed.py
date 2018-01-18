""" Unit tests for pipelines expressed via dask.delayed


"""

import logging
import os
import sys
import unittest

from dask import delayed

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord


from arl.data.polarisation import PolarisationFrame
from arl.graphs.delayed import create_zero_vis_graph_list, create_predict_graph, create_invert_graph, \
    create_deconvolve_graph, create_residual_graph, create_restore_graph
from arl.image.operations import export_image_to_fits, smooth_image, qa_image
from arl.imaging import predict_skycomponent_visibility
from arl.skycomponent.operations import insert_skycomponent
from arl.util.testing_support import create_named_configuration, ingest_unittest_visibility, create_unittest_model, \
    create_unittest_components, insert_unittest_errors
from arl.visibility.operations import qa_visibility

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImagingDelayed(unittest.TestCase):
    def setUp(self):
        self.compute = True
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
    
    def actualSetUp(self, add_errors=False, freqwin=7, block=False, dospectral=True, dopol=False):
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
                                                                   [self.frequency[i]],
                                                                   [self.channelwidth[i]],
                                                                   self.times,
                                                                   self.vis_pol,
                                                                   self.phasecentre, block=block)
                               for i, _ in enumerate(self.frequency)]
        
        self.model_graph = [delayed(create_unittest_model, nout=freqwin)(self.vis_graph_list, self.image_pol,
                                                                         [self.frequency[i]],
                                                                         npixel=self.params['npixel'])
                            for i, _ in enumerate(self.frequency)]
        
        self.components_graph = [delayed(create_unittest_components)(self.model_graph[i], flux[i, :][numpy.newaxis, :])
                                 for i, _ in enumerate(self.frequency)]
        
        # Apply the LOW primary beam and insert into model
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
    
    def _predict_base(self, context='2d', extra=''):
        vis_graph_list = create_zero_vis_graph_list(self.vis_graph_list)
        vis_graph_list = create_predict_graph(vis_graph_list, self.model_graph, context=context, **self.params)
        if self.compute:
            qa = qa_visibility(vis_graph_list[0].compute(), context='predict_%s%s' % (context, extra))
            assert qa.data['maxabs'] > 0.0, str(qa)
    
    def _invert_base(self, context='2d', extra='', flux_max=100.0, flux_min=-0.2, flux_tolerance=5.0):
        dirty_graph = create_invert_graph(self.vis_graph_list, self.model_graph,
                                          context=context,
                                          dopsf=False, normalize=True,
                                          **self.params)
        
        if self.compute:
            dirty = dirty_graph[0].compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_delayed_invert_%s%s_dirty.fits' %
                                 (self.dir, context, extra,))
            qa = qa_image(dirty[0])
            
            assert numpy.abs(qa.data['max'] - flux_max) < flux_tolerance, str(qa)
            assert numpy.abs(qa.data['min'] - flux_min) < flux_tolerance, str(qa)
    
    def test_predict_2d(self):
        self.actualSetUp()
        self._predict_base(context='2d')
    
    def test_predict_facets(self):
        self.actualSetUp()
        self._predict_base(context='facets')
    
    @unittest.skip("Intrinsically unstable")
    def test_predict_facets_wstack(self):
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['padding'] = 8
        self.actualSetUp()
        self._predict_base(context='facets_wstack')
    
    # @unittest.skip("Interpolation insufficently accurate")
    def test_predict_timeslice(self):
        self.actualSetUp()
        self.params['vis_slices'] = self.ntimes
        self._predict_base(context='timeslice')
    
    @unittest.skip("Interpolation insufficently accurate")
    def test_predict_timeslice_wprojection(self):
        self.params['kernel'] = 'wprojection'
        self.actualSetUp()
        self.params['vis_slices'] = self.ntimes
        self._predict_base(context='timeslice', extra='_wprojection')
    
    def test_predict_wprojection(self):
        self.params['kernel'] = 'wprojection'
        self.actualSetUp()
        self._predict_base(context='2d', extra='_wprojection')
    
    def test_predict_wstack(self):
        self.actualSetUp()
        self.params["vis_slices"] = 51
        self._predict_base(context='wstack')
    
    def test_predict_wstack_wprojection(self):
        self.params['kernel'] = 'wprojection'
        self.params["vis_slices"] = 11
        self.actualSetUp()
        self._predict_base(context='wstack', extra='_wprojection')
    
    def test_predict_wstack_spectral(self):
        self.params["vis_slices"] = 11
        self.actualSetUp(dospectral=True)
        self._predict_base(context='wstack', extra='_spectral')
    
    def test_predict_wstack_spectral_pol(self):
        self.params["vis_slices"] = 11
        self.actualSetUp(dospectral=True, dopol=True)
        self._predict_base(context='wstack', extra='_spectral')
    
    def test_invert_2d(self):
        self.actualSetUp()
        self._invert_base(context='2d', flux_max=116.9, flux_min=-3.0, flux_tolerance=3.0)
    
    def test_invert_facets(self):
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['padding'] = 8
        self.actualSetUp()
        self._invert_base(context='facets', flux_max=116.9, flux_min=-3.0, flux_tolerance=3.0)
    
    def test_invert_facets_timeslice(self):
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['padding'] = 4
        self.actualSetUp()
        self.params['vis_slices'] = self.ntimes
        self._invert_base(context='facets_timeslice', flux_max=116.9, flux_min=-3.0, flux_tolerance=3.0)
    
    @unittest.skip("Seems to be correcting twice!")
    def test_invert_facets_wprojection(self):
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['padding'] = 8
        self.actualSetUp()
        self._invert_base(context='facets', extra='_wprojection', flux_max=116.9, flux_min=-3.0, flux_tolerance=3.0)
    
    @unittest.skip("Combination unreliable")
    def test_invert_facets_wstack(self):
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['padding'] = 8
        self.actualSetUp()
        self.params["vis_slices"] = 11
        self._invert_base(context='facets_wstack', flux_max=116.9, flux_min=-3.0, flux_tolerance=3.0)
    
    def test_invert_timeslice(self):
        self.actualSetUp()
        self.params["vis_slices"] = self.ntimes
        self._invert_base(context='timeslice', flux_max=116.9, flux_min=-3.0, flux_tolerance=3.0)
    
    def test_invert_timeslice_wprojection(self):
        self.actualSetUp()
        self.params["vis_slices"] = self.ntimes
        self.params['wstep'] = 4.0
        self.params['kernel'] = 'wprojection'
        self._invert_base(context='timeslice', extra='_wprojection', flux_max=116.9, flux_min=-3.0, flux_tolerance=3.0)
    
    def test_invert_wprojection(self):
        self.params['kernel'] = 'wprojection'
        self.actualSetUp()
        self._invert_base(context='2d', extra='_wprojection', flux_max=116.9, flux_min=-3.0, flux_tolerance=3.0)
    
    def test_invert_wprojection_wstack(self):
        self.params['kernel'] = 'wprojection'
        self.params['vis_slices'] = 7
        self.actualSetUp()
        self._invert_base(context='wstack', extra='_wprojection', flux_max=116.9, flux_min=-3.0, flux_tolerance=3.0)
    
    def test_invert_wstack(self):
        self.params['vis_slices'] = 51
        self.actualSetUp()
        self._invert_base(context='wstack', flux_max=116.9, flux_min=-3.0, flux_tolerance=3.0)
    
    def test_invert_wstack_spectral(self):
        self.params['vis_slices'] = 51
        self.actualSetUp(dospectral=True)
        self._invert_base(context='wstack', extra='_spectral', flux_max=116.9, flux_min=-3.0, flux_tolerance=3.0)
    
    def test_invert_wstack_spectral_pol(self):
        self.params['vis_slices'] = 51
        self.actualSetUp(dospectral=True, dopol=True)
        self._invert_base(context='wstack', extra='_spectral_pol', flux_max=116.9, flux_min=-11.5, flux_tolerance=3.0)
    
    def test_deconvolve_and_restore_cube_spectral(self):
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
        residual_graph = create_residual_graph(self.vis_graph_list, model_graph=dec_graph,
                                               context='wstack', vis_slices=51)
        rest_graph = create_restore_graph(dec_graph, psf_graph, residual_graph)
        restored = rest_graph[0].compute()
        export_image_to_fits(restored, '%s/test_imaging_delayed_restored.fits' % self.dir)
    
    def test_deconvolve_and_restore_cube_mmclean(self):
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
                                            threshold=0.1, nmajor=0, gain=0.7)
        residual_graph = create_residual_graph(self.vis_graph_list, model_graph=dec_graph,
                                               context='wstack', vis_slices=51)
        rest_graph = create_restore_graph(model_graph=dec_graph, psf_graph=psf_graph, residual_graph=residual_graph,
                                          empty=self.model_graph)
        restored = rest_graph[0].compute()
        export_image_to_fits(restored, '%s/test_imaging_delayed_mmclean_restored.fits' % self.dir)


if __name__ == '__main__':
    unittest.main()
