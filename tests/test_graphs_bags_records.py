""" Unit tests for pipelines expressed via dask.bag


"""

import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord
from dask import bag

from arl.calibration.operations import create_gaintable_from_blockvisibility, apply_gaintable
from arl.data.polarisation import PolarisationFrame
from arl.graphs.bags import invert_bag, predict_bag, deconvolve_bag, restore_bag, \
    residual_image_bag, predict_record_subtract, reify, selfcal_bag
from arl.graphs.dask_init import get_dask_Client
from arl.image.operations import qa_image, export_image_to_fits, copy_image, \
    create_empty_image_like
from arl.imaging import create_image_from_visibility, predict_skycomponent_visibility, \
    predict_skycomponent_blockvisibility
from arl.skycomponent.operations import create_skycomponent, insert_skycomponent
from arl.util.testing_support import create_named_configuration
from arl.util.testing_support import simulate_gaintable
from arl.visibility.base import create_visibility, create_blockvisibility


class TestDaskBagsRecords(unittest.TestCase):
    
    def setUp(self):
        
        self.compute = True
        
        if self.compute:
            # Client automatically registers itself as the default scheduler
            self.client = get_dask_Client()
        
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        
        self.npixel = 512
        
        self.setupVis(add_errors=False, block=False)
    
    def setupVis(self, add_errors=False, block=True, freqwin=3):
        self.freqwin = freqwin
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        self.vis_bag = \
            bag.from_sequence([{'freqwin': f,
                                'vis': self.ingest_visibility([freq], times=self.times,
                                                              add_errors=add_errors,
                                                              block=block)}
                               for f, freq in enumerate(self.frequency)])
        
        self.vis_bag = reify(self.vis_bag)
        self.model_bag = bag.from_sequence(self.freqwin * [self.model])
        self.empty_model_bag = bag.from_sequence(self.freqwin * [self.empty_model])
    
    def ingest_visibility(self, freq=[1e8], chan_width=[1e6], times=None, reffrequency=None, add_errors=False,
                          block=True):
        if times is None:
            times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 5)

        if reffrequency is None:
            reffrequency = [1e8]
        lowcore = create_named_configuration('LOWBD2-CORE')
        frequency = numpy.array(freq)
        channel_bandwidth = numpy.array(chan_width)
        
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
        
        export_image_to_fits(model, '%s/test_bags_model.fits' % (self.dir))
        
        if add_errors:
            # These will be the same for all calls
            numpy.random.seed(180555)
            gt = create_gaintable_from_blockvisibility(vt)
            gt = simulate_gaintable(gt, phase_error=1.0, amplitude_error=0.0)
            vt = apply_gaintable(vt, gt)
        return vt
    
    def test_invert_bag(self):
        peaks = {'2d': 65.2997439062, 'timeslice_single': 99.6183393299, 'wstack_single': 100.702701119}
        vis_slices = {'2d': None, 'timeslice_single': 'auto', 'wstack_single': 101}
        for context in ['wstack_single', '2d', 'timeslice_single']:
            dirty_bag = invert_bag(self.vis_bag, self.empty_model, dopsf=False,
                                   context=context, normalize=True,
                                   vis_slices=vis_slices[context])
            dirty, sumwt = dirty_bag.compute()[0]['image']
            export_image_to_fits(dirty, '%s/test_bags_%s_dirty.fits' % (self.dir, context))
            qa = qa_image(dirty, context=context)
            
            assert numpy.abs(qa.data['max'] - peaks[context]) < 1.0e-2, str(qa)
    
    def test_predict_bag(self):
        errors = {'2d': 28.0, 'timeslice_single': 30.0, 'wstack_single': 2.3}
        vis_slices = {'2d': None, 'timeslice_single': 'auto', 'wstack_single': 101}
        for context in ['wstack_single', 'timeslice_single']:
            model_vis_bag = predict_bag(self.vis_bag, self.model, context,
                                        vis_slices=vis_slices[context])
            
            model_vis_bag = reify(model_vis_bag)
            error_vis_bag = self.vis_bag.map(predict_record_subtract, model_vis_bag)
            error_vis_bag = reify(error_vis_bag)
            error_image_bag = invert_bag(error_vis_bag, self.model, dopsf=False,
                                         context=context, normalize=True, vis_slices=vis_slices[context])
            error_image_bag.visualize('test_predict_bag.svg')
            result = error_image_bag.compute()
            error_image = result[0]['image'][0]
            export_image_to_fits(error_image,
                                 '%s/test_bags_%s_predict_error_image.fits' % (self.dir, context))
            qa = qa_image(error_image, context='error image for %s' % context)
            assert qa.data['max'] < errors[context], str(qa)
    
    def test_deconvolve_bag(self):
        context = 'wstack_single'
        vis_slices = {'2d': None, 'timeslice_single': 'auto', 'wstack_single': 101}
        dirty_bag = invert_bag(self.vis_bag, self.model_bag, dopsf=False, context=context,
                               normalize=True,
                               vis_slices=vis_slices[context])
        psf_bag = invert_bag(self.vis_bag, self.model, dopsf=True, context=context,
                             normalize=True,
                             vis_slices=vis_slices[context])
        dirty_bag = reify(dirty_bag)
        psf_bag = reify(psf_bag)
        model_bag = deconvolve_bag(dirty_bag, psf_bag, self.empty_model_bag, niter=1000, gain=0.7,
                                   algorithm='msclean', threshold=0.01, window_shape=None)
        model = model_bag.compute()[0]
        qa = qa_image(model, context=context)
        
        export_image_to_fits(model, '%s/test_bags_%s_deconvolve.fits' % (self.dir, context))
        
        assert numpy.abs(qa.data['max'] - 60.5) < 0.1, str(qa)
    
    def test_restore_bag(self):
        
        peaks = {'wstack_single': 98.8113067286}
        vis_slices = {'wstack_single': 101}
        context = 'wstack_single'
        dirty_bag = invert_bag(self.vis_bag, self.model, dopsf=False, context=context,
                               normalize=True,
                               vis_slices=vis_slices[context])
        psf_bag = invert_bag(self.vis_bag, self.model, dopsf=True, context=context,
                             normalize=True,
                             vis_slices=vis_slices[context])
        dirty_bag = reify(dirty_bag)
        psf_bag = reify(psf_bag)
        model_bag = deconvolve_bag(dirty_bag, psf_bag, self.empty_model_bag, niter=1000, gain=0.7,
                                   algorithm='msclean', threshold=0.01, window_shape=None)
        
        model = model_bag.compute()[0]
        res_image_bag = residual_image_bag(self.vis_bag, model, context=context,
                                           vis_slices=vis_slices[context])
        
        residual = res_image_bag.compute()[0]['image'][0]
        export_image_to_fits(residual, '%s/test_bags_%s_residual.fits' %
                             (self.dir, context))
        
        final_bag = restore_bag(model_bag, psf_bag, res_image_bag)
        final = final_bag.compute()[0]
        qa = qa_image(final, context=context)
        export_image_to_fits(final, '%s/test_bags_%s_restored.fits' %
                             (self.dir, context))
        assert numpy.abs(qa.data['max'] - peaks[context]) < 0.1, str(qa)
    
    def test_residual_image_bag(self):
        context = 'wstack_single'
        vis_slices = {'wstack_single': 101}
        dirty_bag = invert_bag(self.vis_bag, self.empty_model, dopsf=False, context=context,
                               normalize=True, vis_slices=vis_slices[context])
        psf_bag = invert_bag(self.vis_bag, self.empty_model, dopsf=True, context=context,
                             normalize=True, vis_slices=vis_slices[context])
        dirty_bag = reify(dirty_bag)
        psf_bag = reify(psf_bag)
        model_bag = deconvolve_bag(dirty_bag, psf_bag, self.empty_model_bag, niter=1000,
                                   gain=0.1, algorithm='msclean',
                                   threshold=0.01, window_shape=None)
        model = model_bag.compute()[0]
        residual_bag = residual_image_bag(self.vis_bag, model, context=context,
                                          vis_slices=vis_slices[context])
        final = residual_bag.compute()[0]['image'][0]
        export_image_to_fits(final, '%s/test_bags_%s_residual.fits' % (self.dir, context))
        
        qa = qa_image(final, context=context)
        assert qa.data['max'] < 15.0, str(qa)
    
    def test_residual_image_bag_model(self):
        context = 'wstack_single'
        vis_slices = {'wstack_single': 101}
        residual_bag = residual_image_bag(self.vis_bag, self.model, context=context,
                                          vis_slices=vis_slices[context])
        final = residual_bag.compute()[0]['image'][0]
        export_image_to_fits(final, '%s/test_bags_%s_residual_image_bag.fits' % (self.dir, context))
        
        qa = qa_image(final, context=context)
        assert qa.data['max'] < 2.3, str(qa)
    
    @unittest.skip("Global not yet bagged properly")
    def test_selfcal_global_bag(self):
        
        self.setupVis(add_errors=True)
        selfcal_vis_bag = selfcal_bag(self.vis_bag, self.model_bag, global_solution=True,
                                      context='wstack_single', vis_slices=51)
        dirty_bag = invert_bag(selfcal_vis_bag, self.model_bag,
                               dopsf=False, normalize=True, context='wstack_single',
                               vis_slices=101)
        if self.compute:
            dirty, sumwt = dirty_bag.compute()[0]['image']
            export_image_to_fits(dirty, '%s/test_imaging_bags_global_selfcal_dirty.fits'
                                 % (self.dir))
            qa = qa_image(dirty)
            
            assert numpy.abs(qa.data['max'] - 101.7) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 3.5) < 1.0, str(qa)
    
    def test_selfcal_nonglobal_bag(self):
        
        self.setupVis(add_errors=True)
        selfcal_vis_bag = selfcal_bag(self.vis_bag, self.model_bag, global_solution=False,
                                      context='wstack_single', vis_slices=51)
        
        dirty_bag = invert_bag(selfcal_vis_bag, self.model_bag,
                               dopsf=False, normalize=True, context='wstack_single',
                               vis_slices=101)
        if self.compute:
            dirty, sumwt = dirty_bag.compute()[0]['image']
            export_image_to_fits(dirty, '%s/test_imaging_bags_nonglobal_selfcal_dirty.fits' % (self.dir))
            qa = qa_image(dirty)
            
            assert numpy.abs(qa.data['max'] - 100.57) < 0.1, str(qa)
            assert numpy.abs(qa.data['min'] + 4.24) < 0.1, str(qa)
