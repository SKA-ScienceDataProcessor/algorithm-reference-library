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
from arl.graphs.graphs import create_invert_facet_graph, create_predict_facet_graph, \
    create_zero_vis_graph_list, create_subtract_vis_graph_list, \
    create_deconvolve_facet_graph, create_deconvolve_channel_graph, \
    create_invert_wstack_graph, create_residual_wstack_graph, create_predict_wstack_graph, \
    create_predict_graph, create_invert_graph, create_residual_facet_graph, \
    create_residual_facet_wstack_graph, create_predict_facet_wstack_graph, \
    create_invert_facet_wstack_graph, create_invert_timeslice_graph, \
    create_predict_timeslice_graph, create_invert_facet_timeslice_graph, \
    create_predict_facet_timeslice_graph, create_selfcal_graph_list
from arl.graphs.vis import simple_vis
from arl.image.operations import qa_image, export_image_to_fits, copy_image
from arl.imaging import create_image_from_visibility, predict_skycomponent_blockvisibility, \
    invert_wstack_single, predict_wstack_single
from arl.skycomponent.operations import create_skycomponent, insert_skycomponent
from arl.util.testing_support import create_named_configuration, simulate_gaintable
from arl.visibility.base import create_blockvisibility
from arl.visibility.operations import qa_visibility


class TestDaskGraphs(unittest.TestCase):
    def setUp(self):
        
        self.compute = False
        
        self.results_dir = './test_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.invert = invert_wstack_single
        self.predict = predict_wstack_single
        
        self.npixel = 256
        self.facets = 4
        
        self.vis_graph_list = self.setupVis(add_errors=False)
        self.model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2])
    
    def setupVis(self, add_errors=False, freqwin=3):
        self.freqwin = freqwin
        vis_graph_list = list()
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        for freq in self.frequency:
            vis_graph_list.append(delayed(self.ingest_visibility)(freq, times=self.times, add_errors=add_errors))
        
        self.nvis = len(vis_graph_list)
        self.wstep = 10.0
        self.vis_slices = 2 * int(numpy.ceil(numpy.max(numpy.abs(vis_graph_list[0].compute().w)) / self.wstep)) + 1
        return vis_graph_list
    
    def ingest_visibility(self, freq=1e8, chan_width=1e6, times=None, reffrequency=None, add_errors=False):
        if times is None:
            times = [0.0]
        if reffrequency is None:
            reffrequency = [1e8]
        lowcore = create_named_configuration('LOWBD2-CORE')
        frequency = numpy.array([freq])
        channel_bandwidth = numpy.array([chan_width])
        
        #        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        # Observe at zenith to ensure that timeslicing works well. We test that elsewhere.
        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
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
        self.actualmodel = copy_image(model)
        export_image_to_fits(model, '%s/test_imaging_model.fits' % (self.results_dir))
        if add_errors:
            # These will be the same for all calls
            numpy.random.seed(180555)
            gt = create_gaintable_from_blockvisibility(vt)
            gt = simulate_gaintable(gt, phase_error=1.0, amplitude_error=0.0)
            vt = apply_gaintable(vt, gt)
        return vt
    
    def get_LSM(self, vt, cellsize=0.001, reffrequency=None, flux=0.0):
        if reffrequency is None:
            reffrequency = [1e8]
        model = create_image_from_visibility(vt, npixel=self.npixel, cellsize=cellsize, npol=1,
                                             frequency=reffrequency,
                                             polarisation_frame=PolarisationFrame("stokesI"))
        model.data[..., 32, 32] = flux
        return model

    def test_predict_graph(self):
        flux_model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2], flux=100.0)
        zero_vis_graph_list = create_zero_vis_graph_list(self.vis_graph_list)
        predicted_vis_graph_list = create_predict_graph(zero_vis_graph_list, flux_model_graph,
                                                               vis_slices=self.vis_slices)
        residual_vis_graph_list = create_subtract_vis_graph_list(self.vis_graph_list, predicted_vis_graph_list)
        self.compute=True
        if self.compute:
            qa = qa_visibility(self.vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1600.0, 0)
            qa = qa_visibility(predicted_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 100.064844507, 0)
            qa = qa_visibility(residual_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1682.1, 0)
    def test_predict_wstack_graph(self):
        flux_model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2], flux=100.0)
        zero_vis_graph_list = create_zero_vis_graph_list(self.vis_graph_list)
        predicted_vis_graph_list = create_predict_wstack_graph(zero_vis_graph_list, flux_model_graph,
                                                               vis_slices=self.vis_slices)
        residual_vis_graph_list = create_subtract_vis_graph_list(self.vis_graph_list, predicted_vis_graph_list)
        if self.compute:
            qa = qa_visibility(self.vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1600.0, 0)
            qa = qa_visibility(predicted_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 100.064844507, 0)
            qa = qa_visibility(residual_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1629.2, 0)

    def test_predict_timeslice_graph(self):
        flux_model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2], flux=100.0)
        zero_vis_graph_list = create_zero_vis_graph_list(self.vis_graph_list)
        predicted_vis_graph_list = create_predict_timeslice_graph(zero_vis_graph_list, flux_model_graph,
                                                               vis_slices=3)
        residual_vis_graph_list = create_subtract_vis_graph_list(self.vis_graph_list, predicted_vis_graph_list)
        if self.compute:
            qa = qa_visibility(self.vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1600.0, 0)
            qa = qa_visibility(predicted_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 152.1, 0)
            qa = qa_visibility(residual_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1749.2, 0)

    def test_predict_timeslice_graph_wprojection(self):
        flux_model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2], flux=100.0)
        zero_vis_graph_list = create_zero_vis_graph_list(self.vis_graph_list)
        predicted_vis_graph_list = create_predict_timeslice_graph(zero_vis_graph_list, flux_model_graph,
                                                            vis_slices=3, kernel='wprojection',
                                                                  wstep=4.0)
        residual_vis_graph_list = create_subtract_vis_graph_list(self.vis_graph_list, predicted_vis_graph_list)
        if self.compute:
            qa = qa_visibility(self.vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1600.0, 0)
            qa = qa_visibility(predicted_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 165.5, 0)
            qa = qa_visibility(residual_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1709.4, 0)

    def test_predict_facet_wstack_graph(self):
        flux_model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2], flux=100.0)
        zero_vis_graph_list = create_zero_vis_graph_list(self.vis_graph_list)
        predicted_vis_graph_list = create_predict_facet_wstack_graph(zero_vis_graph_list, flux_model_graph,
                                                                     facets=2, vis_slices=self.vis_slices)
        residual_vis_graph_list = create_subtract_vis_graph_list(self.vis_graph_list, predicted_vis_graph_list)
        if self.compute:
            qa = qa_visibility(self.vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1600.0, 0)
            qa = qa_visibility(predicted_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 100.064844507, 0)
            qa = qa_visibility(residual_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1643.1, 0)

    def test_predict_wstack_graph_wprojection(self):
        flux_model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2], flux=100.0)
        zero_vis_graph_list = create_zero_vis_graph_list(self.vis_graph_list)
        predicted_vis_graph_list = create_predict_wstack_graph(zero_vis_graph_list, flux_model_graph,
                                                               vis_slices=11, wstep=10.0, kernel='wprojection')
        residual_vis_graph_list = create_subtract_vis_graph_list(self.vis_graph_list, predicted_vis_graph_list)
        if self.compute:
            qa = qa_visibility(self.vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1600.0, 0)
            qa = qa_visibility(predicted_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 100.064844507, 0)
            qa = qa_visibility(residual_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1668.3018405354974, 0)

    def test_predict_facet_timeslice_graph_wprojection(self):
        flux_model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2], flux=100.0)
        zero_vis_graph_list = create_zero_vis_graph_list(self.vis_graph_list)
        predicted_vis_graph_list = create_predict_facet_timeslice_graph(zero_vis_graph_list, flux_model_graph,
                                                               vis_slices=3, facets=2,
                                                                        wstep=4.0, kernel='wprojection')
        simple_vis(predicted_vis_graph_list[0], filename='predict_facet_timeslice_graph_wprojection', format='png')
        residual_vis_graph_list = create_subtract_vis_graph_list(self.vis_graph_list, predicted_vis_graph_list)
        if self.compute:
            qa = qa_visibility(self.vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1600.0, 0)
            qa = qa_visibility(predicted_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 94.2, 0)
            qa = qa_visibility(residual_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1656.6, 0)

    def test_predict_wprojection_graph(self):
        flux_model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2], flux=100.0)
        zero_vis_graph_list = create_zero_vis_graph_list(self.vis_graph_list)
        predicted_vis_graph_list = create_predict_graph(zero_vis_graph_list, flux_model_graph, wstep=4.0,
                                                        kernel='wprojection')
        residual_vis_graph_list = create_subtract_vis_graph_list(self.vis_graph_list, predicted_vis_graph_list)
        if self.compute:
            qa = qa_visibility(self.vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1600.0, 0)
            qa = qa_visibility(predicted_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 111.8, 0)
            qa = qa_visibility(residual_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1644.3, 0)
    
    def test_predict_facet_graph(self):
        flux_model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2],
                                                 flux=100.0)
        zero_vis_graph_list = create_zero_vis_graph_list(self.vis_graph_list)
        predicted_vis_graph_list = create_predict_facet_graph(zero_vis_graph_list, flux_model_graph,
                                                              facets=self.facets)
        residual_vis_graph_list = create_subtract_vis_graph_list(self.vis_graph_list,
                                                                 predicted_vis_graph_list)
        if self.compute:
            qa = qa_visibility(self.vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1600.0, 0)
        
            qa = qa_visibility(predicted_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 100.064844507, 0)
        
            qa = qa_visibility(residual_vis_graph_list[0].compute())
            numpy.testing.assert_almost_equal(qa.data['maxabs'], 1645.8, 0)

    def test_invert_graph(self):
    
        dirty_graph = create_invert_graph(self.vis_graph_list, self.model_graph,
                                          dopsf=False, normalize=True)
    
        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_invert_graph_dirty.fits' % (self.results_dir))
            qa = qa_image(dirty[0])
        
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)

    def test_invert_graph_wprojection(self):
    
        dirty_graph = create_invert_graph(self.vis_graph_list, self.model_graph,
                                          dopsf=False, normalize=True,
                                          kernel='wprojection', wstep=10.0)
    
        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_invert_graph_wprojection_dirty.fits' % (self.results_dir))
            qa = qa_image(dirty[0])
        
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)

    def test_invert_facet_graph(self):
        
        dirty_graph = create_invert_facet_graph(self.vis_graph_list, self.model_graph,
                                                dopsf=False, normalize=True, facets=4)
        
        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_invert_facet_dirty.fits' % (self.results_dir))
            qa = qa_image(dirty[0])
        
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)

    def test_invert_wstack_graph(self):
    
        dirty_graph = create_invert_wstack_graph(self.vis_graph_list, self.model_graph,
                                                 dopsf=False, normalize=True,
                                                 vis_slices=self.vis_slices)
    
        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_invert_wstack_dirty.fits' % (self.results_dir))
            qa = qa_image(dirty[0])
        
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)

    def test_invert_timeslice_graph_wprojection(self):
    
        dirty_graph = create_invert_timeslice_graph(self.vis_graph_list, self.model_graph,
                                                    dopsf=False, normalize=True,
                                                    vis_slices=3, kernel='wprojection',
                                                    wstep=10.0)
    
        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_invert_timeslice_wprojection_dirty.fits' % (
                self.results_dir))
            qa = qa_image(dirty[0])
        
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)

    def test_invert_timeslice_graph(self):
    
        dirty_graph = create_invert_timeslice_graph(self.vis_graph_list, self.model_graph,
                                                    dopsf=False, normalize=True,
                                                    vis_slices=3)
    
        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_invert_timeslice_dirty.fits' % (self.results_dir))
            qa = qa_image(dirty[0])
        
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)

    def test_invert_facet_wstack_graph(self):
    
        dirty_graph = create_invert_facet_wstack_graph(self.vis_graph_list, self.model_graph,
                                                       dopsf=False, normalize=True,
                                                       vis_slices=self.vis_slices, facets=4)
    
        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_facet_wstack_dirty.fits' % (self.results_dir))
            qa = qa_image(dirty[0])
        
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)

    def test_invert_facet_wstack_graph_wprojection(self):
    
        dirty_graph = create_invert_facet_wstack_graph(self.vis_graph_list, self.model_graph,
                                                       dopsf=False, normalize=True,
                                                       vis_slices=self.vis_slices, facets=2,
                                                       kernel='wprojection', wstep=4.0)
    
        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_invert_facet_wstack_wprojection_dirty.fits' % (
                self.results_dir))
            qa = qa_image(dirty[0])
        
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)

    def test_invert_facet_timeslice_graph(self):
    
        dirty_graph = create_invert_facet_timeslice_graph(self.vis_graph_list, self.model_graph,
                                                          dopsf=False, normalize=True,
                                                          vis_slices=self.vis_slices, facets=2)
    
        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_invert_facet_timeslice_dirty.fits' % (self.results_dir))
            qa = qa_image(dirty[0])
        
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)

    def test_invert_facet_timeslice_graph_wprojection(self):
    
        dirty_graph = create_invert_facet_timeslice_graph(self.vis_graph_list, self.model_graph,
                                                          dopsf=False, normalize=True,
                                                          vis_slices=self.vis_slices, facets=2,
                                                          kernel='wprojection', wstep=4.0)
    
        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_invert_facet_timeslice_wprojection_dirty.fits' % (
                self.results_dir))
            qa = qa_image(dirty[0])
        
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)

    def test_invert_wstack_graph_wprojection(self):
        
        dirty_graph = create_invert_wstack_graph(self.vis_graph_list, self.model_graph,
                                                 dopsf=False, normalize=True, vis_slices=self.vis_slices//10,
                                                 wstep=4.0, facets=2,
                                                 kernel='wprojection')
        
        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_invert_wstack_wprojection_dirty.fits' % (self.results_dir))
            qa = qa_image(dirty[0])
        
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)
    
    def test_residual_facet_graph(self):
        
        self.model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2],
                                                 flux=100.0)
        
        dirty_graph = create_residual_facet_graph(self.vis_graph_list, self.model_graph,
                                                  facets=self.facets)
        
        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_residual_facet%d.fits' %
                             (self.results_dir, self.facets))
        
            qa = qa_image(dirty[0])
        
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)
    
    def test_residual_wstack_graph(self):
        
        self.model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2],
                                                 flux=100.0)
        
        dirty_graph = create_residual_wstack_graph(self.vis_graph_list, self.model_graph,
                                                   vis_slices=self.vis_slices)
        
        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_residual_wstack_slices%d.fits' %
                             (self.results_dir, self.vis_slices))
        
            qa = qa_image(dirty[0])
        
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)

    def test_residual_facet_wstack_graph(self):
    
        self.model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2],
                                                 flux=100.0)
    
        dirty_graph = create_residual_facet_wstack_graph(self.vis_graph_list, self.model_graph,
                                                         facets=4, vis_slices=self.vis_slices)
    
        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_residual_wstack_slices%d.fits' %
                                 (self.results_dir, self.vis_slices))
        
            qa = qa_image(dirty[0])
        
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)

    def test_residual_wstack_graph_wprojection(self):
        
        self.model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2], flux=100.0)
        
        dirty_graph = create_residual_wstack_graph(self.vis_graph_list, self.model_graph,
                                                   kernel='wprojection', vis_slices=11, wstep=10.0)
        
        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_residual_wprojection.fits' %
                             (self.results_dir))
        
            qa = qa_image(dirty[0])
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)
    
    def test_deconvolution_facet_graph(self):
        
        facets = 4
        model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2],
                                            flux=0.0)
        dirty_graph = create_invert_wstack_graph(self.vis_graph_list, model_graph,
                                                 dopsf=False, vis_slices=self.vis_slices)
        psf_model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2],
                                                flux=0.0)
        psf_graph = create_invert_wstack_graph(self.vis_graph_list, psf_model_graph,
                                               vis_slices=self.vis_slices,
                                               dopsf=True)
        
        clean_graph = create_deconvolve_facet_graph(dirty_graph, psf_graph, model_graph,
                                                    algorithm='hogbom', niter=1000,
                                                    fractional_threshold=0.02, threshold=2.0,
                                                    gain=0.1, facets=facets)
        if self.compute:
            result = clean_graph.compute()
        
            export_image_to_fits(result, '%s/test_imaging_deconvolution_facets%d.clean.fits' %
                             (self.results_dir, facets))
        
            qa = qa_image(result)
        
            assert numpy.abs(qa.data['max'] - 100.1) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 1.8) < 1.0, str(qa)

    @unittest.skip("Not yet ready")
    def test_deconvolution_channel_graph(self):
        
        self.vis_graph_list = self.setupVis(freqwin=8)
        self.model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2], frequency=self.frequency)

        model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2],
                                            flux=0.0)
        dirty_graph = create_invert_wstack_graph(self.vis_graph_list, model_graph,
                                                 dopsf=False, vis_slices=self.vis_slices)
        psf_model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2],
                                                flux=0.0)
        psf_graph = create_invert_wstack_graph(self.vis_graph_list, psf_model_graph,
                                               vis_slices=self.vis_slices,
                                               dopsf=True)
    
        channel_images = 4
        clean_graph = create_deconvolve_channel_graph(dirty_graph, psf_graph, model_graph,
                                                    algorithm='hogbom', niter=1000,
                                                    fractional_threshold=0.02, threshold=2.0,
                                                    gain=0.1, subimages=channel_images)
        self.compute = True
        if self.compute:
            result = clean_graph.compute()
        
            export_image_to_fits(result, '%s/test_imaging_deconvolution_channels%d.clean.fits' %
                                 (self.results_dir, channel_images))
        
            qa = qa_image(result)
        
            assert numpy.abs(qa.data['max'] - 100.1) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 1.8) < 1.0, str(qa)

    def test_selfcal_global_graph(self):
    
        corrupted_vis_graph_list = self.setupVis(add_errors=True)
    
        selfcal_vis_graph_list = create_selfcal_graph_list(corrupted_vis_graph_list,
                                                           delayed(self.actualmodel),
                                                           global_solution=True,
                                                           c_predict_graph=create_predict_wstack_graph,
                                                           vis_slices=self.vis_slices)

        dirty_graph = create_invert_wstack_graph(selfcal_vis_graph_list, self.model_graph,
                                                 dopsf=False, normalize=True,
                                                 vis_slices=self.vis_slices)
        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_graphs_global_selfcal_dirty.fits' % (self.results_dir))
            qa = qa_image(dirty[0])
    
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)

    def test_selfcal_nonglobal_graph(self):
    
        corrupted_vis_graph_list = self.setupVis(add_errors=True)
    
        selfcal_vis_graph_list = create_selfcal_graph_list(corrupted_vis_graph_list,
                                                           delayed(self.actualmodel),
                                                           global_solution=False,
                                                           c_predict_graph=create_predict_wstack_graph,
                                                           vis_slices=self.vis_slices)

        dirty_graph = create_invert_wstack_graph(selfcal_vis_graph_list, self.model_graph,
                                                 dopsf=False, normalize=True,
                                                 vis_slices=self.vis_slices)

        if self.compute:
            dirty = dirty_graph.compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_graphs_nonglobal_selfcal_dirty.fits' % (self.results_dir))
            qa = qa_image(dirty[0])
    
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)
