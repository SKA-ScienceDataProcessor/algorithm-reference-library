""" Unit tests for pipelines expressed via dask.bag


"""

import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord
from dask import bag
from distributed import Client

from arl.data.polarisation import PolarisationFrame
from arl.graphs.bags import invert_bag, predict_bag, deconvolve_bag, restore_bag, residual_visibility_bag
from arl.image.operations import qa_image, export_image_to_fits, copy_image
from arl.imaging import create_image_from_visibility, predict_skycomponent_visibility, \
    invert_wstack_single, predict_wstack_single
from arl.skycomponent.operations import create_skycomponent, insert_skycomponent
from arl.util.testing_support import create_named_configuration
from arl.visibility.base import create_visibility
from arl.visibility.operations import qa_visibility

class TestDaskBags(unittest.TestCase):
    def setUp(self):
        
        self.compute = False
        
        self.results_dir = './test_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.invert = invert_wstack_single
        self.predict = predict_wstack_single
        
        self.npixel = 512
        self.facets = 4
        
        self.setupVis(add_errors=False)
    
    def setupVis(self, add_errors=False, freqwin=3):
        self.freqwin = freqwin
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        self.vis_bag=bag.from_sequence([self.ingest_visibility(freq)
                              for freq in numpy.linspace(0.8e8,1.2e8,self.freqwin)])
    
    def ingest_visibility(self, freq=1e8, chan_width=1e6, times=None, reffrequency=None, add_errors=False):
        if times is None:
            times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 5)

        if reffrequency is None:
            reffrequency = [1e8]
        lowcore = create_named_configuration('LOWBD2-CORE')
        frequency = numpy.array([freq])
        channel_bandwidth = numpy.array([chan_width])
        
        #        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        # Observe at zenith to ensure that timeslicing works well. We test that elsewhere.
        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        vt = create_visibility(lowcore, times, frequency, channel_bandwidth=channel_bandwidth,
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
                comp=create_skycomponent(flux=flux, frequency=frequency, direction=sc,
                                                 polarisation_frame=PolarisationFrame("stokesI"))
                comps.append(comp)
        predict_skycomponent_visibility(vt, comps)
        insert_skycomponent(model, comps)
        self.model = copy_image(model)
        export_image_to_fits(model, '%s/test_bags_model.fits' % (self.results_dir))
        return vt

    def test_invert_bag(self):
        c = Client()
        peaks = {'2d': 65.3839320676, 'timeslice': 99.6744590463, 'wstack': 100.670594645}
        vis_slices = {'2d': None, 'timeslice': 'auto', 'wstack': 101}
        model = copy_image(self.model)
        for context in ['wstack', '2d', 'timeslice']:
            dirty_bag = invert_bag(self.vis_bag, model, dopsf=False, context=context, normalize=True,
                                   vis_slices=vis_slices[context])
            future = c.compute(dirty_bag)
            dirty, sumwt = future.result()[0]
            export_image_to_fits(dirty, '%s/test_bag_%s_dirty.fits' % (self.results_dir, context))
            qa = qa_image(dirty, context=context)
        
            assert numpy.abs(qa.data['max'] - peaks[context]) < 1.0e-7, str(qa)

    def test_predict_bag(self):
        c = Client()
        peaks = {'2d': 1599.71314313, 'timeslice': 1666.76383254, 'wstack': 1590.0505978}
        vis_slices = {'2d': None, 'timeslice': 'auto', 'wstack': 101}
        for context in ['2d', 'timeslice', 'wstack']:
            newvis_bag = predict_bag(self.vis_bag, self.model, context, vis_slices=vis_slices[context])
            future = c.compute(newvis_bag)
            newvis = future.result()
            qa = qa_visibility(newvis[0], context=context)
        
            assert numpy.abs(qa.data['maxabs'] - peaks[context]) < 1.0e-7, str(qa)

    def test_deconvolve_bag(self):
        c = Client()
        context = 'wstack'
        vis_slices = {'2d': None, 'timeslice': 'auto', 'wstack': 101}
        dirty_bag = invert_bag(self.vis_bag, self.model, dopsf=False, context=context, normalize=True,
                               vis_slices=vis_slices[context])
        psf_bag = invert_bag(self.vis_bag, self.model, dopsf=True, context=context, normalize=True,
                             vis_slices=vis_slices[context])
        model_bag = deconvolve_bag(dirty_bag, psf_bag, niter=100, gain=1.0, algorithm='hogbom',
                                   threshold=0.01, window_shape=None)
        future = c.compute(model_bag)
        model = future.result()[0]
        qa = qa_image(model, context=context)
    
        export_image_to_fits(model, '%s/test_bag_%s_deconvolve.fits' % (self.results_dir, context))
    
        assert numpy.abs(qa.data['max'] - 100.678376329) < 1.0e-7, str(qa)

    def test_restore_bag(self):
        c = Client()
        context = 'timeslice'
        vis_slices = {'2d': None, 'timeslice': 'auto', 'wstack': 101}
        dirty_bag = invert_bag(self.vis_bag, self.model, dopsf=False, context=context, normalize=True,
                               vis_slices=vis_slices[context])
        psf_bag = invert_bag(self.vis_bag, self.model, dopsf=True, context=context, normalize=True,
                             vis_slices=vis_slices[context])
        model_bag = deconvolve_bag(dirty_bag, psf_bag, niter=1000, gain=0.1, algorithm='msclean',
                                   threshold=0.01, window_shape=None)
        
        model_vis_bag=predict_bag(self.vis_bag, model_bag, context=context, vis_slices=vis_slices[context])
        residual_vis_bag=residual_visibility_bag(self.vis_bag, model_vis_bag)

        future = c.compute(model_bag)
        model = future.result()[0]
        export_image_to_fits(model, '%s/test_bag_%s_deconvolve.fits' % (self.results_dir, context))

        residual_bag=invert_bag(residual_vis_bag, model_bag, dopsf=False, context=context, normalize=True)
        future = c.compute(residual_bag)
        final = future.result()[0][0]
        export_image_to_fits(final, '%s/test_bag_%s_residual.fits' % (self.results_dir, context))

        final_bag=restore_bag(model_bag, psf_bag, residual_bag)
        future = c.compute(final_bag)
        final = future.result()[0]
        qa = qa_image(final, context=context)
        export_image_to_fits(final, '%s/test_bag_%s_restored.fits' % (self.results_dir, context))
    
        assert numpy.abs(qa.data['max'] - 100.309563954) < 1.0e-7, str(qa)




