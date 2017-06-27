"""Unit tests for pipelines expressed via dask.delayed


"""

import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord
from dask import delayed

from arl.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility
from arl.data.polarisation import PolarisationFrame
from arl.fourier_transforms.ftprocessor import create_image_from_visibility, predict_skycomponent_blockvisibility, \
    invert_wprojection, predict_wprojection
from arl.graphs.dask_graphs import create_invert_facet_graph, create_predict_facet_graph, \
    create_zero_vis_graph_list, create_subtract_vis_graph_list, create_continuum_imaging_pipeline_graph, \
    create_ical_pipeline_graph, create_deconvolve_facet_graph, create_invert_graph, \
    create_residual_graph, create_residual_facet_graph
from arl.image.operations import qa_image, export_image_to_fits
from arl.skycomponent.operations import create_skycomponent, insert_skycomponent
from arl.util.testing_support import create_named_configuration, simulate_gaintable
from arl.visibility.operations import create_blockvisibility
from arl.visibility.operations import qa_visibility


class TestImagingDask(unittest.TestCase):
    def setUp(self):
        
        self.results_dir = './results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # In[3]:
        
        self.setupVis(add_errors=False)
        self.npixel = 256
        self.facets = 2
        self.model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2], npixel=self.npixel)
    
    def tearDown(self):
        pass
    
    def setupVis(self, add_errors=False):
        self.freqwin = 3
        self.vis_graph_list = list()
        for freq in numpy.linspace(0.8e8, 1.2e8, self.freqwin):
            self.vis_graph_list.append(delayed(self.ingest_visibility)(freq, add_errors=add_errors))
    
        self.nvis = len(self.vis_graph_list)

    def ingest_visibility(self, freq=1e8, chan_width=1e6, time=0.0, reffrequency=[1e8], add_errors=False):
        lowcore = create_named_configuration('LOWBD2-CORE')
        times = [time]
        frequency = numpy.array([freq])
        channel_bandwidth = numpy.array([chan_width])
        
#        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox=2000.0)
        # Observe at zenith to ensure that timeslicing works well. We test that elsewhere.
        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-26.7 * u.deg, frame='icrs', equinox=2000.0)
        vt = create_blockvisibility(lowcore, times, frequency, channel_bandwidth=channel_bandwidth,
                                    weight=1.0, phasecentre=phasecentre,
                                    polarisation_frame=PolarisationFrame("stokesI"))
        cellsize = 0.001
        model = create_image_from_visibility(vt, npixel=self.npixel, cellsize=cellsize, npol=1,
                                             frequency=reffrequency,
                                             polarisation_frame=PolarisationFrame("stokesI"))
        flux = numpy.array([[100.0]])
        facets = 4
        
        rpix = model.wcs.wcs.crpix - 1
        spacing_pixels = self.npixel // facets
        centers = [-1.5, -0.5, 0.5, 1.5]
        comps = list()
        for iy in centers:
            for ix in centers:
                p = int(round(rpix[0] + ix * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[0]))), \
                    int(round(rpix[1] + iy * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[1])))
                sc = pixel_to_skycoord(p[0], p[1], model.wcs, origin=0)
                comps.append(create_skycomponent(flux=flux, frequency=vt.frequency, direction=sc,
                                                 polarisation_frame=PolarisationFrame("stokesI")))
        predict_skycomponent_blockvisibility(vt, comps)
        insert_skycomponent(model, comps)
        export_image_to_fits(model, '%s/test_imaging_dask_model.fits' % (self.results_dir))

        if add_errors:
            gt = create_gaintable_from_blockvisibility(vt)
            gt = simulate_gaintable(gt, phase_error=1.0, amplitude_error=0.0)
            vt = apply_gaintable(vt, gt)
        return vt
    
    def get_LSM(self, vt, npixel=512, cellsize=0.001, reffrequency=[1e8], flux=0.0):
        model = create_image_from_visibility(vt, npixel=self.npixel, cellsize=cellsize, npol=1,
                                             frequency=reffrequency,
                                             polarisation_frame=PolarisationFrame("stokesI"))
        model.data[...,32,32]=flux
        return model
    
    def test_predict_facet(self):
        flux_model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2], npixel=self.npixel,
                                                 flux=100.0)
        zero_vis_graph_list = create_zero_vis_graph_list(self.vis_graph_list)
        predicted_vis_graph_list = create_predict_facet_graph(zero_vis_graph_list, flux_model_graph,
                                                              invert_single=invert_wprojection,
                                                              predict_single=predict_wprojection,
                                                              wstep=25,
                                                              facets=self.facets)
        residual_vis_graph_list = create_subtract_vis_graph_list(self.vis_graph_list,
                                                                 predicted_vis_graph_list)
        
        qa = qa_visibility(self.vis_graph_list[0].compute())
        numpy.testing.assert_almost_equal(qa.data['maxabs'], 1600.0, 0)
        
        qa = qa_visibility(predicted_vis_graph_list[0].compute())
        numpy.testing.assert_almost_equal(qa.data['maxabs'], 98.690192366452763, 0)
        
        qa = qa_visibility(residual_vis_graph_list[0].compute())
        numpy.testing.assert_almost_equal(qa.data['maxabs'], 1614.0773242619071, 0)
    
    def test_invert_facet(self):
        
        qa_bench = None
        for facets in [1, 2, 4]:
            dirty_graph = create_invert_facet_graph(self.vis_graph_list, self.model_graph,
                                                    dopsf=False, normalize=True,
                                                    invert_single=invert_wprojection,
                                                    wstep=25,
                                                    facets=facets, padding=2 * facets)
            
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_dask_invert_dirty_facets%d.fits' %
                                 (self.results_dir, facets))
            qa = qa_image(dirty[0])
            print(qa)

            if qa_bench is None:
                qa_bench = qa
                assert numpy.abs(qa_bench.data['max']-100.0) < 1.0
                assert numpy.abs(qa_bench.data['min']+5.0) < 1.0

            assert numpy.abs(qa.data['max'] - qa_bench.data['max']) < 1.0
            assert numpy.abs(qa.data['min'] - qa_bench.data['min']) < 1.0
    
    def test_residual_facet(self):

        qa_bench = None
        self.model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2],
                                                 npixel=self.npixel, flux=100.0)
        for facets in [1, 2, 4]:
            dirty_graph = create_residual_facet_graph(self.vis_graph_list, self.model_graph, facets=facets,
                                                      predict_single=predict_wprojection,
                                                      invert_single=invert_wprojection,
                                                      wstep=25,
                                                      padding=2 * facets)
            
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_dask_residual_residual_facets%d.fits' %
                                 (self.results_dir, facets))

            qa = qa_image(dirty[0])
            print(qa)

            if qa_bench is None:
                qa_bench = qa
                assert numpy.abs(qa_bench.data['max'] - 100.0) < 5.0
                assert numpy.abs(qa_bench.data['min'] + 5.0) < 5.0

            assert numpy.abs(qa.data['max'] - qa_bench.data['max']) < 5.0
            assert numpy.abs(qa.data['min'] - qa_bench.data['min']) < 5.0

    def test_deconvolution_facet(self):
        
        qa_bench = None
        for facets in [1, 2, 4]:
            model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2],
                                                npixel=self.npixel, flux=0.0)
            dirty_graph = create_invert_facet_graph(self.vis_graph_list, model_graph,
                                                    invert_single=invert_wprojection,
                                                    wstep=25,
                                                    dopsf=False)
            
            export_image_to_fits(dirty_graph.compute()[0],
                                 '%s/test_imaging_dask_deconvolution_facets%d.dirty.fits' %
                                 (self.results_dir, facets))
            psf_model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2],
                                                    npixel=self.npixel // facets, flux=0.0)
            psf_graph = create_invert_graph(self.vis_graph_list, psf_model_graph,
                                            invert_single=invert_wprojection,
                                            wstep=25,
                                            dopsf=True)
            export_image_to_fits(psf_graph.compute()[0],
                                 '%s/test_imaging_dask_deconvolution_facets%d.psf.fits' %
                                 (self.results_dir, facets))

            clean_graph = create_deconvolve_facet_graph(dirty_graph, psf_graph, model_graph,
                                                        algorithm='hogbom', niter=1000,
                                                        fractional_threshold=0.02, threshold=2.0,
                                                        gain=0.1, facets=facets)
            result = clean_graph.compute()
            
            export_image_to_fits(result, '%s/test_imaging_dask_deconvolution_facets%d.clean.fits' %
                                 (self.results_dir, facets))

            qa = qa_image(result)
            print(qa)

            if qa_bench is None:
                qa_bench = qa
                assert numpy.abs(qa_bench.data['max'] - 97.0) < 5.0
                assert numpy.abs(qa_bench.data['min'] + 2.0) < 5.0

            assert numpy.abs(qa.data['max'] - qa_bench.data['max']) < 5.0
            assert numpy.abs(qa.data['min'] - qa_bench.data['min']) < 5.0

    def test_continuum_imaging_pipeline(self):
        continuum_imaging_graph = \
            create_continuum_imaging_pipeline_graph(self.vis_graph_list, model_graph=self.model_graph,
                                                    c_deconvolve_graph=create_deconvolve_facet_graph,
                                                    c_invert_graph=create_invert_graph,
                                                    c_residual_graph=create_residual_graph,
                                                    predict_single=predict_wprojection,
                                                    invert_single=invert_wprojection,
                                                    facets=2,
                                                    niter=1000, fractional_threshold=0.1,
                                                    wstep=25,
                                                    threshold=2.0, nmajor=3, gain=0.1)
        clean, residual, restored = continuum_imaging_graph.compute()
        export_image_to_fits(clean, '%s/test_imaging_dask_continuum_imaging_pipeline_clean.fits' % (self.results_dir))
        export_image_to_fits(residual[0], '%s/test_imaging_dask_continuum_imaging_pipeline_residual.fits' % (self.results_dir))
        export_image_to_fits(restored, '%s/test_imaging_dask_continuum_imaging_pipeline_restored.fits' % (self.results_dir))

        qa = qa_image(restored)
        print(qa)
        assert numpy.abs(qa.data['max'] - 100.0) < 5.0
        assert numpy.abs(qa.data['min'] + 1.3) < 5.0


    def test_ical_pipeline(self):
        self.setupVis(add_errors=True)
        ical_graph = \
            create_ical_pipeline_graph(self.vis_graph_list, model_graph=self.model_graph,
                                       c_deconvolve_graph=create_deconvolve_facet_graph,
                                       c_invert_graph=create_invert_graph,
                                       c_residual_graph=create_residual_graph,
                                       predict_single=predict_wprojection,
                                       invert_single=invert_wprojection, facets=2,
                                       niter=1000, fractional_threshold=0.1,
                                       threshold=2.0, nmajor=2,
                                       wstep=25,
                                       gain=0.1, first_selfcal=1)
        clean, residual, restored = ical_graph.compute()
        export_image_to_fits(clean, '%s/test_imaging_dask_ical_pipeline_clean.fits' % (self.results_dir))
        export_image_to_fits(residual[0], '%s/test_imaging_dask_ical_pipeline_residual.fits' % (self.results_dir))
        export_image_to_fits(restored, '%s/test_imaging_dask_ical_pipeline_restored.fits' % (self.results_dir))

        qa = qa_image(restored)
        print(qa)
        assert numpy.abs(qa.data['max'] - 100.0) < 5.0
        assert numpy.abs(qa.data['min'] + 1.3) < 5.0