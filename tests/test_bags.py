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
from arl.graphs.bags import invert_bag, predict_bag, deconvolve_bag, restore_bag, residual_image_bag,\
    residual_vis_bag, qa_visibility_bag
from arl.image.operations import qa_image, export_image_to_fits, copy_image
from arl.imaging import create_image_from_visibility, predict_skycomponent_visibility
from arl.skycomponent.operations import create_skycomponent, insert_skycomponent
from arl.util.testing_support import create_named_configuration
from arl.visibility.base import create_visibility,copy_visibility
from arl.visibility.operations import qa_visibility


class TestDaskBags(unittest.TestCase):
    def setUp(self):
        
        self.compute = False
        
        self.results_dir = './test_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.npixel = 512
        
        self.setupVis(add_errors=False)
    
    def setupVis(self, add_errors=False, freqwin=3):
        self.freqwin = freqwin
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        self.vis_bag = bag.from_sequence([self.ingest_visibility(freq)
                                          for freq in numpy.linspace(0.8e8, 1.2e8, self.freqwin)])
    
    def ingest_visibility(self, freq=1e8, chan_width=1e6, times=None, reffrequency=None, add_errors=False):
        if times is None:
            times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 5)
        
        if reffrequency is None:
            reffrequency = [1e8]
        lowcore = create_named_configuration('LOWBD2-CORE')
        frequency = numpy.array([freq])
        channel_bandwidth = numpy.array([chan_width])
        
        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        vt = create_visibility(lowcore, times, frequency, channel_bandwidth=channel_bandwidth,
                               weight=1.0, phasecentre=phasecentre,
                               polarisation_frame=PolarisationFrame("stokesI"))
        cellsize = 0.001
        model = create_image_from_visibility(vt, npixel=self.npixel, cellsize=cellsize, npol=1,
                                             frequency=reffrequency, phasecentre=phasecentre,
                                             polarisation_frame=PolarisationFrame("stokesI"))
        flux = numpy.array([[100.0]])
        facets = 4
        
        rpix = model.wcs.wcs.crpix
        spacing_pixels = self.npixel // facets
        centers = [-1.5, -0.5, 0.5, 1.5]
        comps = list()
        for iy in centers:
            for ix in centers:
                p = int(round(rpix[0] + ix * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[0]))), \
                    int(round(rpix[1] + iy * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[1])))
                sc = pixel_to_skycoord(p[0], p[1], model.wcs, origin=0)
                comp = create_skycomponent(flux=flux, frequency=frequency, direction=sc,
                                           polarisation_frame=PolarisationFrame("stokesI"))
                comps.append(comp)
        predict_skycomponent_visibility(vt, comps)
        insert_skycomponent(model, comps)
        self.model = copy_image(model)
        export_image_to_fits(model, '%s/test_bags_model.fits' % (self.results_dir))
        return vt
    
    def test_invert_bag(self):
        peaks = {'2d': 65.440798589, 'timeslice': 99.7403479215, 'wstack': 100.654001673}
        vis_slices = {'2d': None, 'timeslice': 'auto', 'wstack': 101}
        model = copy_image(self.model)
        for context in ['wstack', '2d', 'timeslice']:
            dirty_bag = invert_bag(self.vis_bag, model, dopsf=False, context=context, normalize=True,
                                   vis_slices=vis_slices[context])
            dirty, sumwt = list(dirty_bag)[0]
            export_image_to_fits(dirty, '%s/test_bag_%s_dirty.fits' % (self.results_dir, context))
            qa = qa_image(dirty, context=context)
            
            assert numpy.abs(qa.data['max'] - peaks[context]) < 1.0e-7, str(qa)
    
    def test_predict_bag(self):
        errors = {'2d': 28.0, 'timeslice': 31.0, 'wstack': 2.3}
        vis_slices = {'2d': None, 'timeslice': 'auto', 'wstack': 101}
        for context in ['2d', 'timeslice', 'wstack']:
            model_vis_bag = predict_bag(self.vis_bag, self.model, context, vis_slices=vis_slices[context])
            error_vis_bag = residual_vis_bag(self.vis_bag, model_vis_bag)
            error_image_bag=invert_bag(error_vis_bag, self.model, dopsf=False, context=context, normalize=True,
                                   vis_slices=vis_slices[context])
            result=list(error_image_bag)[0]
            error_image=result[0]
            export_image_to_fits(error_image, '%s/test_bag_%s_predict_error_image.fits' % (self.results_dir, context))
            qa = qa_image(error_image, context='error image for %s' % context)
            assert qa.data['max'] < errors[context], str(qa)

    def test_deconvolve_bag(self):
        context = 'wstack'
        vis_slices = {'2d': None, 'timeslice': 'auto', 'wstack': 101}
        dirty_bag = invert_bag(self.vis_bag, self.model, dopsf=False, context=context, normalize=True,
                               vis_slices=vis_slices[context])
        psf_bag = invert_bag(self.vis_bag, self.model, dopsf=True, context=context, normalize=True,
                             vis_slices=vis_slices[context])
        model_bag = deconvolve_bag(dirty_bag, psf_bag, niter=1000, gain=0.7, algorithm='msclean',
                                   threshold=0.01, window_shape=None)
        model = list(model_bag)[0]
        qa = qa_image(model, context=context)
        
        export_image_to_fits(model, '%s/test_bag_%s_deconvolve.fits' % (self.results_dir, context))
        
        assert numpy.abs(qa.data['max'] - 60.5293158864) < 1.0e-7, str(qa)
    
    def test_restore_bag(self):
        context = 'wstack'
        vis_slices = {'2d': None, 'timeslice': 'auto', 'wstack': 101}
        dirty_bag = invert_bag(self.vis_bag, self.model, dopsf=False, context=context, normalize=True,
                               vis_slices=vis_slices[context])
        psf_bag = invert_bag(self.vis_bag, self.model, dopsf=True, context=context, normalize=True,
                             vis_slices=vis_slices[context])
        model_bag = deconvolve_bag(dirty_bag, psf_bag, niter=1000, gain=0.7, algorithm='msclean',
                                   threshold=0.01, window_shape=None)

        residual_bag = residual_image_bag(self.vis_bag, model_bag, context=context, vis_slices=vis_slices[context])
        residual = list(residual_bag)[0][0]
        qa = qa_image(residual, context=context)
        export_image_to_fits(residual, '%s/test_bag_%s_residual.fits' % (self.results_dir, context))
        assert qa.data['max'] < 6.0, str(qa)
        
        final_bag = restore_bag(model_bag, psf_bag, residual_bag)
        final = list(final_bag)[0]
        qa = qa_image(final, context=context)
        export_image_to_fits(final, '%s/test_bag_%s_restored.fits' % (self.results_dir, context))
        assert numpy.abs(qa.data['max'] - 98.7939512789) < 1.0e-7, str(qa)

    def test_residual_image_bag(self):
        context = 'wstack'
        vis_slices = {'2d': None, 'timeslice': 'auto', 'wstack': 101}
        dirty_bag = invert_bag(self.vis_bag, self.model, dopsf=False, context=context, normalize=True,
                               vis_slices=vis_slices[context])
        psf_bag = invert_bag(self.vis_bag, self.model, dopsf=True, context=context, normalize=True,
                             vis_slices=vis_slices[context])
        model_bag = deconvolve_bag(dirty_bag, psf_bag, niter=1000, gain=0.1, algorithm='msclean',
                                   threshold=0.01, window_shape=None)
    
        residual_bag = residual_image_bag(self.vis_bag, model_bag, context=context, vis_slices=vis_slices[context])
        final = list(residual_bag)[0][0]
        export_image_to_fits(final, '%s/test_bag_%s_residual.fits' % (self.results_dir, context))
    
        qa = qa_image(final, context=context)
        assert qa.data['max'] < 15.0, str(qa)


    def test_residual_image_bag_model(self):
        context = 'wstack'
        vis_slices = {'2d': None, 'timeslice': 'auto', 'wstack': 101}
        residual_bag = residual_image_bag(self.vis_bag, self.model, context=context, vis_slices=vis_slices[context])
        final = list(residual_bag)[0][0]
        export_image_to_fits(final, '%s/test_bag_%s_residual_image_bag.fits' % (self.results_dir, context))
    
        qa = qa_image(final, context=context)
        assert qa.data['max'] < 2.3, str(qa)

    def test_residual_vis_bag(self):
        # Test if the subtraction of visibilities works
        context = 'test_residual_vis_bag'
        new_vis_bag=self.vis_bag.map(copy_visibility)
        for inplace in [True, False]:
            residual_bag = residual_vis_bag(self.vis_bag, new_vis_bag)
            final = list(residual_bag)
            qa = qa_visibility(final[0], context="%s %s" % (context, str(inplace)))
            assert numpy.abs(qa.data['maxabs']) < 1.0e-7, str(qa)

