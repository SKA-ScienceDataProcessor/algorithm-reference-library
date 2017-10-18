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
from arl.graphs.graphs import create_deconvolve_facet_graph, create_invert_wstack_graph, \
    create_residual_wstack_graph, create_selfcal_graph_list
from arl.image.operations import qa_image, export_image_to_fits
from arl.imaging import create_image_from_visibility, predict_skycomponent_blockvisibility, \
    invert_wstack_single, predict_wstack_single
from arl.pipelines.graphs import create_continuum_imaging_pipeline_graph, \
    create_ical_pipeline_graph
from arl.skycomponent.operations import create_skycomponent, insert_skycomponent
from arl.util.testing_support import create_named_configuration, simulate_gaintable
from arl.visibility.base import create_blockvisibility


class TestImagingDask(unittest.TestCase):
    def setUp(self):
        
        self.compute = False
        
        self.results_dir = './test_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.invert = invert_wstack_single
        self.predict = predict_wstack_single
        
        self.npixel = 256
        self.facets = 4
        
        self.setupVis(add_errors=False)
        self.model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2], npixel=self.npixel)
    
    def setupVis(self, add_errors=False):
        self.freqwin = 3
        self.vis_graph_list = list()
        for freq in numpy.linspace(0.8e8, 1.2e8, self.freqwin):
            self.vis_graph_list.append(delayed(self.ingest_visibility)(freq, add_errors=add_errors))
        
        self.nvis = len(self.vis_graph_list)
        self.wstep = 10.0
        self.vis_slices = 2 * int(numpy.ceil(numpy.max(numpy.abs(self.vis_graph_list[0].compute().w)) / self.wstep)) + 1
    
    def ingest_visibility(self, freq=1e8, chan_width=1e6, time=0.0, reffrequency=None, add_errors=False):
        if reffrequency is None:
            reffrequency = [1e8]
        lowcore = create_named_configuration('LOWBD2-CORE')
        times = [time]
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
        export_image_to_fits(model, '%s/test_pipelines_model.fits' % (self.results_dir))
        
        if add_errors:
            # These will be the same for all calls
            numpy.random.seed(180555)
            gt = create_gaintable_from_blockvisibility(vt)
            gt = simulate_gaintable(gt, phase_error=1.0, amplitude_error=0.0)
            vt = apply_gaintable(vt, gt)
        return vt
    
    def get_LSM(self, vt, npixel=512, cellsize=0.001, reffrequency=None, flux=0.0):
        if reffrequency is None:
            reffrequency = [1e8]
        model = create_image_from_visibility(vt, npixel=self.npixel, cellsize=cellsize, npol=1,
                                             frequency=reffrequency,
                                             polarisation_frame=PolarisationFrame("stokesI"))
        model.data[..., 32, 32] = flux
        return model

    def test_continuum_imaging_pipeline(self):
        continuum_imaging_graph = \
            create_continuum_imaging_pipeline_graph(self.vis_graph_list, model_graph=self.model_graph,
                                                    c_deconvolve_graph=create_deconvolve_facet_graph,
                                                    c_invert_graph=create_invert_wstack_graph,
                                                    c_residual_graph=create_residual_wstack_graph,
                                                    vis_slices=self.vis_slices, facets=2,
                                                    niter=1000, fractional_threshold=0.1,
                                                    threshold=2.0, nmajor=3, gain=0.1)
        if self.compute:
            clean, residual, restored = continuum_imaging_graph.compute()
            export_image_to_fits(clean, '%s/test_pipelines_continuum_imaging_pipeline_clean.fits' % (self.results_dir))
            export_image_to_fits(residual[0],
                                 '%s/test_pipelines_continuum_imaging_pipeline_residual.fits' % (self.results_dir))
            export_image_to_fits(restored,
                                 '%s/test_pipelines_continuum_imaging_pipeline_restored.fits' % (self.results_dir))
            
            qa = qa_image(restored)
            assert numpy.abs(qa.data['max'] - 100.0) < 5.0, str(qa)
            assert numpy.abs(qa.data['min'] + 5.0) < 5.0, str(qa)
    
    def test_ical_pipeline(self):
        self.setupVis(add_errors=True)
        ical_graph = \
            create_ical_pipeline_graph(self.vis_graph_list, model_graph=self.model_graph,
                                       c_deconvolve_graph=create_deconvolve_facet_graph,
                                       c_invert_graph=create_invert_wstack_graph,
                                       c_residual_graph=create_residual_wstack_graph,
                                       c_selfcal_graph=create_selfcal_graph_list,
                                       global_solution=True,
                                       vis_slices=self.vis_slices, facets=2,
                                       niter=1000, fractional_threshold=0.1,
                                       threshold=2.0, nmajor=5, gain=0.1, first_selfcal=1)
        if self.compute:
            clean, residual, restored = ical_graph.compute()
            export_image_to_fits(clean, '%s/test_pipelines_ical_pipeline_clean.fits' % (self.results_dir))
            export_image_to_fits(residual[0], '%s/test_pipelines_ical_pipeline_residual.fits' % (self.results_dir))
            export_image_to_fits(restored, '%s/test_pipelines_ical_pipeline_restored.fits' % (self.results_dir))
            
            qa = qa_image(restored)
            assert numpy.abs(qa.data['max'] - 100.0) < 5.0, str(qa)
            assert numpy.abs(qa.data['min'] + 5.0) < 5.0, str(qa)
