"""Unit tests for Fourier transform processors


"""
import logging
import unittest
from functools import partial

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord

from arl.data.polarisation import PolarisationFrame
from arl.image.operations import export_image_to_fits, create_empty_image_like, smooth_image
from arl.imaging import predict_2d, predict_wstack, predict_wprojection, predict_facets, \
    predict_timeslice, predict_wprojection_wstack, invert_wprojection_wstack, \
    invert_2d, invert_wstack, invert_wprojection, invert_facets, invert_timeslice, \
    create_image_from_visibility, predict_skycomponent_visibility, \
    predict_facets_wstack, invert_facets_wstack, \
    predict_facets_wprojection, invert_facets_wprojection
from arl.imaging.weighting import weight_visibility
from arl.skycomponent.operations import create_skycomponent, find_skycomponents, find_nearest_component, \
    insert_skycomponent
from arl.util.testing_support import create_named_configuration
from arl.visibility.base import create_visibility
from arl.visibility.operations import sum_visibility

log = logging.getLogger(__name__)

class TestImaging(unittest.TestCase):
    def _checkdirty(self, vis, name='test_invert_2d_dirty', fluxthreshold=1.0):
        # Make the dirty image
        self.params['imaginary'] = False
        dirty = create_empty_image_like(self.model)
        dirty, sumwt = invert_2d(vis=vis, im=dirty, dopsf=False, normalize=True, **self.params)
        export_image_to_fits(dirty, '%s/%s_dirty.fits' % (self.dir, name))
        maxabs = numpy.max(numpy.abs(dirty.data))
        assert maxabs < fluxthreshold, "%s, abs max %f exceeds flux threshold" % (name, maxabs)
    
    def _checkcomponents(self, dirty, fluxthreshold=5.0, positionthreshold=1.0):
        comps = find_skycomponents(dirty, fwhm=1.0, threshold=fluxthreshold, npixels=5)
        assert len(comps) == len(self.components), "Different number of components found: original %d, recovered %d" % \
                                                   (len(self.components), len(comps))
        cellsize = abs(dirty.wcs.wcs.cdelt[0])
        # Check for agreement between image and DFT
        for comp in comps:
            sflux = sum_visibility(self.componentvis, comp.direction)[0]
            assert abs(comp.flux[0, 0] - sflux[0, 0]) < fluxthreshold, \
                "Fitted and DFT flux differ %s %s" % (comp.flux[0, 0], sflux[0, 0])
            # Check for agreement in direction
            ocomp = find_nearest_component(comp.direction, self.components)
            radiff = abs(comp.direction.ra.deg - ocomp.direction.ra.deg) / cellsize
            assert radiff < positionthreshold, "Component differs in dec %.3f pixels" % radiff
            decdiff = abs(comp.direction.dec.deg - ocomp.direction.dec.deg) / cellsize
            assert decdiff < positionthreshold, "Component differs in dec %.3f pixels" % decdiff
    
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
                       'timeslice': 1000.0}
    
    def actualSetUp(self, time=None, frequency=None, dospectral=False, dopol=False):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 5)
        
        if time is not None:
            self.times = time
        log.info("Times are %s" % (self.times))
        
        if dospectral:
            self.nchan=3
            self.frequency = numpy.array([0.9e8, 1e8, 1.1e8])
            self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        else:
            self.frequency = numpy.array([1e8])
            self.channel_bandwidth = numpy.array([1e7])
            
        if dopol:
            self.vis_pol = PolarisationFrame('linear')
            self.image_pol = PolarisationFrame('stokesIQUV')
        else:
            self.vis_pol = PolarisationFrame('stokesI')
            self.image_pol = PolarisationFrame('stokesI')

        if dopol:
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        else:
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
        self.componentvis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_image_from_visibility(self.componentvis, npixel=512, cellsize=0.001,
                                                  nchan=len(self.frequency),
                                                  polarisation_frame=self.image_pol)
        
        # Fill the visibility with exactly computed point sources. These are chosen to lie
        # on grid points.
        spacing_pixels = 512 // 8
        log.info('Spacing in pixels = %s' % spacing_pixels)
        
        centers = [(x, x) for x in numpy.linspace(-3.0,+3.0,7)]

        for x in numpy.linspace(-3.0,+3.0,7):
            centers.append((-x, x))
            
        centers.append((1e-7,1e-7))
        centers.append((1.1, 2.2))

        # Make the list of components
        rpix = self.model.wcs.wcs.crpix - 1
        self.components = []
        for center in centers:
            ix, iy = center
            # The phase center in 0-relative coordinates is n // 2 so we centre the grid of
            # components on ny // 2, nx // 2. The wcs must be defined consistently.
            p = int(round(rpix[0] + ix * spacing_pixels * numpy.sign(self.model.wcs.wcs.cdelt[0]))), \
                int(round(rpix[1] + iy * spacing_pixels * numpy.sign(self.model.wcs.wcs.cdelt[1])))
            sc = pixel_to_skycoord(p[0], p[1], self.model.wcs, origin=0)
            log.info("Component at (%f, %f) [0-rel] %s" % (p[0], p[1], str(sc)))
            
            if ix != 0 and iy != 0:
                
                # Channel images
                comp = create_skycomponent(flux=flux, frequency=self.frequency, direction=sc,
                                           polarisation_frame=self.image_pol)
                self.components.append(comp)
        
        # Predict the visibility from the components exactly.
        self.componentvis.data['vis'] *= 0.0
        predict_skycomponent_visibility(self.componentvis, self.components)
        insert_skycomponent(self.model, self.components)
        
        # Calculate the model convolved with a Gaussian.
        self.cmodel = smooth_image(self.model)
        export_image_to_fits(self.model, '%s/test_model.fits' % self.dir)
        export_image_to_fits(self.cmodel, '%s/test_cmodel.fits' % self.dir)

    def test_findcomponents(self):
        # Check that the components are where we expected them to be after insertion
        self.actualSetUp()
        self._checkcomponents(self.cmodel)

    def test_findcomponents_spectral_pol(self):
        # Check that the components are where we expected them to be after insertion
        self.actualSetUp(dospectral=True, dopol=True)
        self._checkcomponents(self.cmodel)

    def test_predict_2d(self):
        # Test if the 2D prediction works
        #
        # Set w=0 so that the two-dimensional transform should agree exactly with the component transform.
        # Good check on the grid correction in the image->vis direction
        # Set all w to zero
        self.actualSetUp()
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency, channel_bandwidth= \
            self.channel_bandwidth, phasecentre=self.phasecentre, weight=1.0)
        self.componentvis.data['uvw'][:, 2] = 0.0
        # Predict the visibility using direct evaluation
        predict_skycomponent_visibility(self.componentvis, self.components)
        
        self.modelvis = create_visibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                          weight=1.0, polarisation_frame=self.vis_pol)
        self.modelvis.data['uvw'][:, 2] = 0.0
        predict_2d(self.modelvis, self.model, **self.params)
        self.residualvis = create_visibility(self.lowcore, self.times, self.frequency,
                                             channel_bandwidth=self.channel_bandwidth,
                                             phasecentre=self.phasecentre,
                                             weight=1.0, polarisation_frame=self.vis_pol)
        self.residualvis.data['uvw'][:, 2] = 0.0
        self.residualvis.data['vis'] = self.modelvis.data['vis'] - self.componentvis.data['vis']
        
        self._checkdirty(self.residualvis, 'test_predict_2d', fluxthreshold=4.0)
    
    def _predict_base(self, predict, fluxthreshold=1.0):
        self.modelvis = create_visibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                          weight=1.0, polarisation_frame=self.vis_pol)
        self.modelvis.data['vis'] *= 0.0
        predict(self.modelvis, self.model, **self.params)
        self.residualvis = create_visibility(self.lowcore, self.times, self.frequency,
                                             channel_bandwidth=self.channel_bandwidth,
                                             phasecentre=self.phasecentre,
                                             weight=1.0, polarisation_frame=self.vis_pol)
        self.residualvis.data['uvw'][:, 2] = 0.0
        self.residualvis.data['vis'] = self.modelvis.data['vis'] - self.componentvis.data['vis']
        self._checkdirty(self.residualvis, 'test_%s' % predict.__name__, fluxthreshold=fluxthreshold)
    
    def test_predict_facets(self):
        self.actualSetUp()
        self.params['facets'] = 2
        self._predict_base(predict_facets, fluxthreshold=numpy.infty)
    def test_predict_timeslice(self):
        # This works poorly because of the poor interpolation accuracy for point sources. The corresponding
        # invert works well particularly if the beam sampling is high
        self.actualSetUp()
        self._predict_base(predict_timeslice, fluxthreshold=numpy.infty)

    def test_predict_timeslice_wprojection(self):
        predict_timeslice_wprojection = partial(predict_timeslice, predict=predict_wprojection)
        self.actualSetUp()
        self.params['kernel']='wprojection'
        self.params['wstep'] = 2.0
        self._predict_base(predict_timeslice, fluxthreshold=numpy.infty)

    def test_predict_wstack(self):
        self.actualSetUp()
        self.params['wstack'] = 2.0
        self._predict_base(predict_wstack, fluxthreshold=5.0)

    def test_predict_facets_wstack(self):
        self.actualSetUp()
        self.params['wstack'] = 2.0
        self.params['facets'] = 2
        self._predict_base(predict_facets_wstack, fluxthreshold=5.6)

    def test_predict_facets_wstack_spectral(self):
        self.actualSetUp(dospectral=True)
        self.params['wstack'] = 2.0
        self.params['facets'] = 2
        self._predict_base(predict_facets_wstack, fluxthreshold=5.8)

    def test_predict_facets_wstack_spectral_pol(self):
        self.actualSetUp(dospectral=True, dopol=True)
        self.params['wstack'] = 2.0
        self.params['facets'] = 2
        self._predict_base(predict_facets_wstack, fluxthreshold=5.8)

    def test_predict_wstack_wprojection(self):
        self.actualSetUp()
        self.params['wstack'] = 5 * 2.0
        self.params['wstep'] = 2.0
        self._predict_base(predict_wprojection_wstack, fluxthreshold=4.4)

    def test_predict_facets_wprojection(self):
        self.actualSetUp()
        self.params['wstep'] = 2.0
        self.params['facets'] = 2
        self._predict_base(predict_facets_wprojection, fluxthreshold=7.0)

    def test_predict_wprojection(self):
        self.actualSetUp()
        self.params['wstep'] = 2.0
        self._predict_base(predict_wprojection, fluxthreshold=4.0)
    
    def test_invert_2d(self):
        # Test if the 2D invert works with w set to zero
        # Set w=0 so that the two-dimensional transform should agree exactly with the model.
        # Good check on the grid correction in the vis->image direction
        
        self.actualSetUp()
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency,
                                              channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                              weight=1.0, polarisation_frame=self.vis_pol)
        self.componentvis.data['uvw'][:, 2] = 0.0
        self.componentvis.data['vis'] *= 0.0
        # Predict the visibility using direct evaluation
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)
        
        dirty2d = create_empty_image_like(self.model)
        dirty2d, sumwt = invert_2d(self.componentvis, dirty2d, **self.params)
        
        export_image_to_fits(dirty2d, '%s/test_invert_2d_dirty.fits' % self.dir)
        
        self._checkcomponents(dirty2d, fluxthreshold=20.0, positionthreshold=1.0)
    
    def _invert_base(self, invert, fluxthreshold=20.0, positionthreshold=1.0, check_components=True):
        dirty = create_empty_image_like(self.model)
        dirty, sumwt = invert(self.componentvis, dirty, **self.params)
        assert sumwt.all() > 0.0
        export_image_to_fits(dirty, '%s/test_%s_dirty.fits' % (self.dir, invert.__name__))
        if check_components:
            self._checkcomponents(dirty, fluxthreshold, positionthreshold)

    def test_invert_facets(self):
        self.actualSetUp()
        self.params['facets'] = 2
        self._invert_base(invert_facets, positionthreshold=6.0, check_components=False)

    def test_invert_facets_wprojection(self):
        self.actualSetUp()
        self.params['facets'] = 2
        self.params['wstep'] = 4.0
        self._invert_base(invert_facets_wprojection, positionthreshold=1.0)

    def test_invert_wstack(self):
        self.actualSetUp()
        self.params['wstack'] = 4.0
        self._invert_base(invert_wstack, positionthreshold=1.0)

    def test_invert_wstack_spectral(self):
        self.actualSetUp(dospectral=True)
        self.params['wstack'] = 4.0
        self._invert_base(invert_wstack, positionthreshold=1.0)

    def test_invert_wstack_spectral_pol(self):
        self.actualSetUp(dospectral=True, dopol=True)
        self.params['wstack'] = 4.0
        self._invert_base(invert_wstack, positionthreshold=1.0)

    def test_invert_facets_wstack(self):
        self.actualSetUp()
        self.params['wstack'] = 4.0
        self.params['facets'] = 4
        self._invert_base(invert_facets_wstack, positionthreshold=1.0)

    def test_invert_wprojection_wstack(self):
        self.actualSetUp()
        self.params['wstack'] = 5 * 4.0
        self.params['wstep'] = 4.0
        self._invert_base(invert_wprojection_wstack, positionthreshold=1.0)
    
    def test_invert_wprojection(self):
        self.actualSetUp()
        self.params['wstep'] = 4.0
        self._invert_base(invert_wprojection, positionthreshold=1.0)
    
    def test_invert_timeslice(self):
        self.actualSetUp()
        self._invert_base(invert_timeslice, positionthreshold=8.0, check_components=False)
    
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
