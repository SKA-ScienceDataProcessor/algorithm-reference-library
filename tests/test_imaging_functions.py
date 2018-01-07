""" Unit tests for Fourier transform processors


"""
import logging
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord

from arl.data.polarisation import PolarisationFrame
from arl.image.operations import export_image_to_fits, create_empty_image_like, smooth_image
from arl.imaging import predict_2d, invert_2d, \
    create_image_from_visibility, predict_skycomponent_visibility
from arl.imaging.imaging_context import predict_context, invert_context
from arl.imaging.weighting import weight_visibility
from arl.skycomponent.operations import create_skycomponent, find_skycomponents, find_nearest_component, \
    insert_skycomponent, apply_beam_to_skycomponent
from arl.util.testing_support import create_named_configuration, create_low_test_beam
from arl.visibility.base import create_visibility
from arl.visibility.operations import sum_visibility

log = logging.getLogger(__name__)


class TestImaging(unittest.TestCase):
    def _checkdirty(self, vis, context, fluxthreshold=0.3):
        # Make the dirty image
        self.params['imaginary'] = False
        self.params['timeslice'] = 'auto'
        
        dirty = create_empty_image_like(self.model)
        dirty, sumwt = invert_context(vis=vis, im=dirty, dopsf=False, normalize=True, context='2d', **self.params)
        export_image_to_fits(dirty, '%s/test_imaging_functions_%s_dirty.fits' % (self.dir, context))
        maxabs = numpy.max(numpy.abs(dirty.data))
        assert maxabs < fluxthreshold, "%s, abs max %f exceeds flux threshold" % (context, maxabs)
    
    def _checkcomponents(self, dirty, fluxthreshold=1.0, positionthreshold=1.0, check_dft=False):
        comps = find_skycomponents(dirty, fwhm=1.0, threshold=10*fluxthreshold, npixels=5)
        assert len(comps) == len(self.components), "Different number of components found: original %d, recovered %d" % \
                                                   (len(self.components), len(comps))
        cellsize = abs(dirty.wcs.wcs.cdelt[0])
        
        for comp in comps:
            # Check for agreement in direction
            ocomp = find_nearest_component(comp.direction, self.components)
            radiff = abs(comp.direction.ra.deg - ocomp.direction.ra.deg) / cellsize
            assert radiff < positionthreshold, "Component differs in dec %.3f pixels" % radiff
            decdiff = abs(comp.direction.dec.deg - ocomp.direction.dec.deg) / cellsize
            assert decdiff < positionthreshold, "Component differs in dec %.3f pixels" % decdiff
        
        # Check for agreement between in flux image and DFT
        if check_dft:
            for comp in comps:
                sflux = sum_visibility(self.componentvis, comp.direction)[0]
                assert abs(comp.flux[0, 0] - sflux[0, 0]) < fluxthreshold, \
                    "Fitted and DFT flux differ %s %s" % (comp.flux[0, 0], sflux[0, 0])
    
    def setUp(self):
        import os
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        self.params = {'npixel': 512,
                       'nchan': 1,
                       'reffrequency': 1e8,
                       'facets': 8,
                       'padding': 2,
                       'oversampling': 2,
                       'kernel': '2d',
                       'wstep': 4.0,
                       'wstack': 4.0}
    
    def actualSetUp(self, time=None, dospectral=False, dopol=False):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 5)
        
        if time is not None:
            self.times = time
        log.info("Times are %s" % self.times)
        
        if dospectral:
            self.nchan = 3
            self.frequency = numpy.array([0.9e8, 1e8, 1.1e8])
            self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        else:
            self.frequency = numpy.array([1e8])
            self.channel_bandwidth = numpy.array([1e7])
        
        if dopol:
            self.vis_pol = PolarisationFrame('linear')
            self.image_pol = PolarisationFrame('stokesIQUV')
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        else:
            self.vis_pol = PolarisationFrame('stokesI')
            self.image_pol = PolarisationFrame('stokesI')
            f = numpy.array([100.0])
        
        if dospectral:
            flux = numpy.array([f, 0.8 * f, 0.6 * f])
        else:
            flux = numpy.array([f])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency,
                                              channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                              weight=1.0, polarisation_frame=self.vis_pol)
        self.uvw = self.componentvis.data['uvw']
        self.componentvis.data['vis'][...] = 0.0
        
        # Create model
        self.model = create_image_from_visibility(self.componentvis, npixel=self.params['npixel'], cellsize=0.0005,
                                                  nchan=len(self.frequency),
                                                  polarisation_frame=self.image_pol)
        
        # Fill the visibility with exactly computed point sources.
        spacing_pixels = 512 // 8
        log.info('Spacing in pixels = %s' % spacing_pixels)
        
        centers = [(x, x) for x in numpy.linspace(-1.4, +1.4, 9)]
        
        for x in numpy.linspace(-1.4, +1.4, 9):
            centers.append((-x, x))
        
        centers.append((0.5, 1.1))
        centers.append((1e-7, 1e-7))
        
        # Make the list of components
        rpix = self.model.wcs.wcs.crpix
        self.components = []
        for center in centers:
            ix, iy = center
            # The phase center in 0-relative coordinates is n // 2 so we centre the grid of
            # components on ny // 2, nx // 2. The wcs must be defined consistently.
            p = int(round(rpix[0] + ix * spacing_pixels * numpy.sign(self.model.wcs.wcs.cdelt[0]))), \
                int(round(rpix[1] + iy * spacing_pixels * numpy.sign(self.model.wcs.wcs.cdelt[1])))
            sc = pixel_to_skycoord(p[0], p[1], self.model.wcs, origin=1)
            log.info("Component at (%f, %f) [0-rel] %s" % (p[0], p[1], str(sc)))
            
            if ix != 0 and iy != 0:
                # Channel images
                comp = create_skycomponent(flux=flux, frequency=self.frequency, direction=sc,
                                           polarisation_frame=self.image_pol)
                self.components.append(comp)
        
        # Predict the visibility from the components exactly
        self.componentvis.data['vis'] *= 0.0
        self.beam = create_low_test_beam(self.model)
        
        self.components = apply_beam_to_skycomponent(self.components, self.beam)
        self.componentvis = predict_skycomponent_visibility(self.componentvis, self.components)
        self.model = insert_skycomponent(self.model, self.components)
        
        # Calculate the model convolved with a Gaussian.
        self.cmodel = smooth_image(self.model)
        export_image_to_fits(self.model,  '%s/test_imaging_functions_model.fits' % self.dir)
        export_image_to_fits(self.cmodel, '%s/test_imaging_functions_cmodel.fits' % self.dir)
    
    def test_findcomponents(self):
        # Check that the components are where we expected them to be after insertion
        self.actualSetUp()
        self._checkcomponents(self.cmodel, check_dft=False)
    
    def test_findcomponents_spectral_pol(self):
        # Check that the components are where we expected them to be after insertion
        self.actualSetUp(dospectral=True, dopol=True)
        self._checkcomponents(self.cmodel, check_dft=False)
    
    def test_predict_2d(self):
        # Test if the 2D prediction works
        #
        # Set w=0 so that the two-dimensional transform should agree exactly with the component transform.
        # Good check on the grid correction in the image->vis direction
        # Set all w to zero
        self.actualSetUp()
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency,
                                              channel_bandwidth=self.channel_bandwidth,
                                              phasecentre=self.phasecentre, weight=1.0)
        self.componentvis.data['uvw'][:, 2] = 0.0
        # Predict the visibility using direct evaluation
        self.componentvis = predict_skycomponent_visibility(self.componentvis, self.components)
        
        self.modelvis = create_visibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                          weight=1.0, polarisation_frame=self.vis_pol)
        self.modelvis.data['uvw'][:, 2] = 0.0
        self.modelvis = predict_2d(self.modelvis, self.model, **self.params)
        self.residualvis = create_visibility(self.lowcore, self.times, self.frequency,
                                             channel_bandwidth=self.channel_bandwidth,
                                             phasecentre=self.phasecentre,
                                             weight=1.0, polarisation_frame=self.vis_pol)
        self.residualvis.data['uvw'][:, 2] = 0.0
        self.residualvis.data['vis'] = self.modelvis.data['vis'] - self.componentvis.data['vis']
        
        self._checkdirty(self.residualvis, 'predict_2d')
    
    def _predict_base(self, context='2d', extra='', fluxthreshold=1.0):
        self.modelvis = create_visibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                          weight=1.0, polarisation_frame=self.vis_pol)
        self.modelvis.data['vis'] *= 0.0
        self.modelvis = predict_context(self.modelvis, self.model, context=context, **self.params)
        self.residualvis = create_visibility(self.lowcore, self.times, self.frequency,
                                             channel_bandwidth=self.channel_bandwidth,
                                             phasecentre=self.phasecentre,
                                             weight=1.0, polarisation_frame=self.vis_pol)
        self.residualvis.data['uvw'][:, 2] = 0.0
        self.residualvis.data['vis'] = self.modelvis.data['vis'] - self.componentvis.data['vis']
        self._checkdirty(self.residualvis, 'predict_%s%s' % (context, extra),
                         fluxthreshold=fluxthreshold)
    
    def _invert_base(self, context, extra='', fluxthreshold=1.0, positionthreshold=1.0, check_components=True):
        dirty = create_empty_image_like(self.model)
        dirty, sumwt = invert_context(self.componentvis, dirty, dopsf=False, context=context, **self.params)
        export_image_to_fits(dirty, '%s/test_imaging_functions_invert_%s%s_dirty.fits' % (self.dir, context, extra))
        if check_components:
            self._checkcomponents(dirty, fluxthreshold, positionthreshold)
    
    def test_invert_2d(self):
        # Test if the 2D invert works with w set to zero
        # Set w=0 so that the two-dimensional transform should agree exactly with the model.
        # Good check on the grid correction in the vis->image direction
        
        self.actualSetUp()
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency,
                                              channel_bandwidth=self.channel_bandwidth,
                                              phasecentre=self.phasecentre,
                                              weight=1.0, polarisation_frame=self.vis_pol)
        self.componentvis.data['uvw'][:, 2] = 0.0
        self.componentvis.data['vis'] *= 0.0
        # Predict the visibility using direct evaluation
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)
        
        dirty2d = create_empty_image_like(self.model)
        dirty2d, sumwt = invert_2d(self.componentvis, dirty2d, **self.params)
        
        export_image_to_fits(dirty2d, '%s/test_imaging_functions_invert_2d_dirty.fits' % self.dir)
        
        self._checkcomponents(dirty2d)
    
    def test_psf_location_2d(self):
        
        self.actualSetUp()
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency,
                                              channel_bandwidth=self.channel_bandwidth,
                                              phasecentre=self.phasecentre,
                                              weight=1.0, polarisation_frame=self.vis_pol)
        self.componentvis.data['uvw'][:, 2] = 0.0
        self.componentvis.data['vis'] *= 0.0
        
        psf2d = create_empty_image_like(self.model)
        psf2d, sumwt = invert_2d(self.componentvis, psf2d, dopsf=True, **self.params)
        
        export_image_to_fits(psf2d, '%s/test_imaging_functions_invert_psf_location.fits' % self.dir)
        
        nchan, npol, ny, nx = psf2d.shape
        
        assert numpy.abs(psf2d.data[0, 0, ny // 2, nx // 2] - 1.0) < 2e-3
        imagecentre = pixel_to_skycoord(nx // 2 + 1.0, ny // 2 + 1.0, wcs=psf2d.wcs, origin=1)
        assert imagecentre.separation(self.phasecentre).value < 1e-15, \
            "Image phase centre %s not as expected %s" % (imagecentre, self.phasecentre)
    
    @unittest.skip("Insufficiently accurate")
    def test_predict_facets(self):
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['padding'] = 8
        self.actualSetUp()
        self._predict_base(context='facets')
    
    @unittest.skip("Facetting unreliable")
    def test_predict_facets_timeslice(self):
        self.params['facets'] = 5
        self.params['npixel'] = 5 * 128
        self.params['padding'] = 8
        self.params['timeslice'] = 1e5
        self.actualSetUp()
        self._predict_base(context='facets_timeslice')
    
    @unittest.skip("Facetting unreliable")
    def test_predict_facets_wprojection(self):
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['padding'] = 8
        self.params['kernel'] = 'wprojection'
        self.actualSetUp()
        self._predict_base(context='facets', extra='_wprojection')
    
    @unittest.skip("Combinations unreliable")
    def test_predict_facets_wstack(self):
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['padding'] = 8
        self.actualSetUp()
        self._predict_base(context='facets_wstack')
    
    @unittest.skip("Interpolation insufficently accurate")
    def test_predict_timeslice(self):
        self.params['timeslice'] = 'auto'
        self.actualSetUp()
        self._predict_base(context='timeslice')
    
    @unittest.skip("Interpolation insufficently accurate")
    def test_predict_timeslice_wprojection(self):
        self.params['kernel'] = 'wprojection'
        self.params['timeslice'] = None
        self.actualSetUp()
        self._predict_base(context='timeslice', extra='_wprojection')
    
    def test_predict_wprojection(self):
        self.params['kernel'] = 'wprojection'
        self.actualSetUp()
        self._predict_base(context='2d', extra='_wprojection')
    
    def test_predict_wstack(self):
        self.actualSetUp()
        self._predict_base(context='wstack')
    
    def test_predict_wstack_wprojection(self):
        self.params['kernel'] = 'wprojection'
        self.params['wstack'] = 5 * self.params['wstep']
        self.actualSetUp()
        self._predict_base(context='wstack', extra='_wprojection')
    
    def test_predict_wstack_spectral(self):
        self.actualSetUp(dospectral=True)
        self._predict_base(context='wstack', extra='_spectral', fluxthreshold=7.0)
    
    def test_predict_wstack_spectral_pol(self):
        self.actualSetUp(dospectral=True, dopol=True)
        self._predict_base(context='wstack', extra='_spectral_pol', fluxthreshold=7.0)
    
    def test_invert_facets(self):
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['padding'] = 8
        self.actualSetUp()
        self._invert_base(context='facets', positionthreshold=1.0, check_components=True)
    
    def test_invert_facets_timeslice(self):
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['padding'] = 8
        self.params['timeslice'] = None
        self.actualSetUp()
        self._invert_base(context='facets_timeslice', check_components=True,
                          positionthreshold=2.0)
    
    @unittest.skip("Combination unreliable")
    def test_invert_facets_wprojection(self):
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['padding'] = 8
        self.actualSetUp()
        self._invert_base(context='facets', extra='_wprojection', check_components=True,
                          positionthreshold=2.0)
    
    @unittest.skip("Combination unreliable")
    def test_invert_facets_wstack(self):
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['padding'] = 8
        self.actualSetUp()
        self._invert_base(context='facets_wstack', positionthreshold=1.0, check_components=False)
    
    def test_invert_timeslice(self):
        self.actualSetUp()
        self._invert_base(context='timeslice', positionthreshold=1.0, check_components=True)
    
    def test_invert_timeslice_wprojection(self):
        self.params['kernel'] = 'wprojection'
        self.params['timeslice'] = None
        self.actualSetUp()
        self._invert_base(context='timeslice', extra='_wprojection', positionthreshold=1.0,
                          check_components=True)
    
    def test_invert_wprojection(self):
        self.params['kernel'] = 'wprojection'
        self.actualSetUp()
        self._invert_base(context='2d', extra='_wprojection', positionthreshold=1.0)
    
    def test_invert_wprojection_wstack(self):
        self.params['kernel'] = 'wprojection'
        self.params['wstack'] = 5 * self.params['wstep']
        self.actualSetUp()
        self._invert_base(context='wstack', extra='_wprojection', positionthreshold=1.0)
    
    def test_invert_wstack(self):
        self.params['wstack'] = 1.0
        self.actualSetUp()
        self._invert_base(context='wstack', positionthreshold=2.0)
    
    def test_invert_wstack_spectral(self):
        self.actualSetUp(dospectral=True)
        self._invert_base(context='wstack', extra='_spectral', positionthreshold=2.0)
    
    def test_invert_wstack_spectral_pol(self):
        self.actualSetUp(dospectral=True, dopol=True)
        self._invert_base(context='wstack', extra='_spectral_pol', positionthreshold=2.0)
    
    def test_weighting(self):
        self.actualSetUp()
        vis, density, densitygrid = weight_visibility(self.componentvis, self.model, weighting='uniform')
        assert vis.nvis == self.componentvis.nvis
        assert len(density) == vis.nvis
        assert numpy.std(vis.imaging_weight) > 0.0
        assert densitygrid.data.shape == self.model.data.shape
        vis, density, densitygrid = weight_visibility(self.componentvis, self.model, weighting='natural')
        assert density is None
        assert densitygrid is None
    
    def test_create_image_from_visibility(self):
        self.actualSetUp()
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency,
                                              phasecentre=self.phasecentre, weight=1.0,
                                              polarisation_frame=self.vis_pol,
                                              channel_bandwidth=self.channel_bandwidth)
        im = create_image_from_visibility(self.componentvis, nchan=1, npixel=128)
        assert im.data.shape == (1, 1, 128, 128)
        im = create_image_from_visibility(self.componentvis, frequency=self.frequency, npixel=128)
        assert im.data.shape == (len(self.frequency), 1, 128, 128)
        im = create_image_from_visibility(self.componentvis, frequency=self.frequency, npixel=128,
                                          nchan=1)
        assert im.data.shape == (1, 1, 128, 128)


if __name__ == '__main__':
    unittest.main()
