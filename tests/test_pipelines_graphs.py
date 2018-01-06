"""Unit tests for pipelines expressed via dask.delayed


"""

import logging
import os
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord

from arl.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility
from arl.data.polarisation import PolarisationFrame
from arl.image.operations import qa_image, export_image_to_fits, copy_image, create_empty_image_like
from arl.imaging import create_image_from_visibility, predict_skycomponent_blockvisibility, \
    predict_skycomponent_visibility
from arl.skycomponent.operations import create_skycomponent, insert_skycomponent
from arl.util.testing_support import create_named_configuration
from arl.util.testing_support import simulate_gaintable
from arl.visibility.base import create_visibility, create_blockvisibility

from dask import delayed
from arl.graphs.delayed import create_deconvolve_facet_graph, create_invert_graph, \
    create_residual_graph, create_predict_graph, create_selfcal_graph_list
from arl.pipelines.graphs import create_continuum_imaging_pipeline_graph, \
    create_ical_pipeline_graph

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))

class TestPipelineGraphs(unittest.TestCase):
    def setUp(self):
    
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
    
        self.compute = False
        
        self.npixel = 256
        self.facets = 4
        self.vis_slices = 51
        
        self.setupVis(add_errors=False)
        self.model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2], npixel=self.npixel)
    
    def setupVis(self, add_errors=False, block=True, freqwin=3):
        self.freqwin = freqwin
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)

        self.vis_graph_list = list()
        for freq in numpy.linspace(0.8e8, 1.2e8, self.freqwin):
            self.vis_graph_list.append(delayed(self.ingest_visibility)(freq, add_errors=add_errors))
        
        self.nvis = len(self.vis_graph_list)
   
    def ingest_visibility(self, freq=[1e8], chan_width=[1e6], times=None, reffrequency=None, add_errors=False,
                          block=True):
        if times is None:
            times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 5)
        
        if reffrequency is None:
            reffrequency = [1e8]
        lowcore = create_named_configuration('LOWBD2-CORE')
        ntimes = 5
        times = numpy.linspace(-numpy.pi / 3.0, numpy.pi / 3.0, ntimes)
        frequency = numpy.array([freq])
        channel_bandwidth = numpy.array([chan_width])
        
        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        if block:
            vt = create_blockvisibility(lowcore, times, frequency, channel_bandwidth=channel_bandwidth,
                                        weight=1.0, phasecentre=phasecentre,
                                        polarisation_frame=PolarisationFrame("stokesI"))
        else:
            vt = create_visibility(lowcore, times, frequency, channel_bandwidth=channel_bandwidth,
                                        weight=1.0, phasecentre=phasecentre,
                                        polarisation_frame=PolarisationFrame("stokesI"))
        cellsize = 0.001
        model = create_image_from_visibility(vt, npixel=self.npixel, cellsize=cellsize, npol=1,
                                             frequency=reffrequency, phasecentre=phasecentre,
                                             polarisation_frame=PolarisationFrame("stokesI"))
        flux = numpy.array([[100.0]])
        facets = 4
        
        rpix = model.wcs.wcs.crpix - 1.0
        spacing_pixels = self.npixel // facets
        centers = [-1.5, -0.5, 0.5, 1.5]
        comps = list()
        for iy in centers:
            for ix in centers:
                p = int(round(rpix[0] + ix * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[0]))), \
                    int(round(rpix[1] + iy * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[1])))
                sc = pixel_to_skycoord(p[0], p[1], model.wcs, origin=1)
                comp = create_skycomponent(flux=flux, frequency=frequency, direction=sc,
                                           polarisation_frame=PolarisationFrame("stokesI"))
                comps.append(comp)
        if block:
            predict_skycomponent_blockvisibility(vt, comps)
        else:
            predict_skycomponent_visibility(vt, comps)
        insert_skycomponent(model, comps)
        self.model = copy_image(model)
        self.empty_model = create_empty_image_like(model)
        export_image_to_fits(model, '%s/test_pipeline_bags_model.fits' % (self.dir))
        
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
                                                     threshold=2.0, nmajor=0, gain=0.1)
        if self.compute:
            clean, residual, restored = continuum_imaging_graph.compute()
            export_image_to_fits(clean[0], '%s/test_pipelines_continuum_imaging_pipeline_clean.fits' % (self.dir))
            export_image_to_fits(residual[0],
                                 '%s/test_pipelines_continuum_imaging_pipeline_residual.fits' % (self.dir))
            export_image_to_fits(restored,
                                 '%s/test_pipelines_continuum_imaging_pipeline_restored.fits' % (self.dir))
            
            qa = qa_image(restored)
            assert numpy.abs(qa.data['max'] - 100.0) < 5.0, str(qa)
            assert numpy.abs(qa.data['min'] + 5.0) < 5.0, str(qa)

    def test_ical_pipeline(self):
        self.setupVis(add_errors=True)
        ical_graph = \
            create_ical_pipeline_graph(self.vis_graph_list, model_graph=self.model_graph,
                                       c_deconvolve_graph=create_deconvolve_facet_graph,
                                       c_predict_graph=create_predict_wstack_graph,
                                       c_invert_graph=create_invert_wstack_graph,
                                       c_residual_graph=create_residual_wstack_graph,
                                       c_selfcal_graph=create_selfcal_graph_list,
                                       global_solution=False,
                                       vis_slices=self.vis_slices, facets=2,
                                       niter=1000, fractional_threshold=0.1,
                                       threshold=2.0, nmajor=5, gain=0.1, first_selfcal=1)
        if self.compute:
            clean, residual, restored = ical_graph.compute()
            export_image_to_fits(clean, '%s/test_pipelines_ical_pipeline_clean.fits' % (self.dir))
            export_image_to_fits(residual[0], '%s/test_pipelines_ical_pipeline_residual.fits' % (self.dir))
            export_image_to_fits(restored, '%s/test_pipelines_ical_pipeline_restored.fits' % (self.dir))
        
            qa = qa_image(restored)
            assert numpy.abs(qa.data['max'] - 100.0) < 5.0, str(qa)
            assert numpy.abs(qa.data['min'] + 5.0) < 5.0, str(qa)

    def test_ical_pipeline_global(self):
        self.setupVis(add_errors=True)
        ical_graph = \
            create_ical_pipeline_graph(self.vis_graph_list, model_graph=self.model_graph,
                                       c_deconvolve_graph=create_deconvolve_facet_graph,
                                       c_predict_graph=create_predict_wstack_graph,
                                       c_invert_graph=create_invert_wstack_graph,
                                       c_residual_graph=create_residual_wstack_graph,
                                       c_selfcal_graph=create_selfcal_graph_list,
                                       global_solution=True,
                                       vis_slices=self.vis_slices, facets=2,
                                       niter=1000, fractional_threshold=0.1,
                                       threshold=2.0, nmajor=5, gain=0.1, first_selfcal=1)
        if self.compute:
            clean, residual, restored = ical_graph.compute()
            export_image_to_fits(clean, '%s/test_pipelines_ical_pipeline_global_clean.fits' % (self.dir))
            export_image_to_fits(residual[0], '%s/test_pipelines_ical_pipeline_global_residual.fits' % (self.dir))
            export_image_to_fits(restored, '%s/test_pipelines_ical_pipeline_global_restored.fits' % (self.dir))
        
            qa = qa_image(restored)
            assert numpy.abs(qa.data['max'] - 100.0) < 5.0, str(qa)
            assert numpy.abs(qa.data['min'] + 5.0) < 5.0, str(qa)

if __name__ == '__main__':
    unittest.main()