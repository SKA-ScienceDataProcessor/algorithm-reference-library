""" Unit tests for Fourier transform processors


"""
import logging
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord

from arl.data.polarisation import PolarisationFrame
from arl.image.operations import export_image_to_fits, create_empty_image_like, smooth_image, qa_image
from arl.imaging import predict_2d, invert_2d, create_image_from_visibility, predict_skycomponent_visibility
from arl.imaging.imaging_context import predict_function, invert_function
from arl.imaging.weighting import weight_visibility
from arl.skycomponent.operations import find_skycomponents, find_nearest_component, insert_skycomponent
from arl.util.testing_support import create_named_configuration, ingest_unittest_visibility, create_unittest_model, \
    insert_unittest_errors, create_unittest_components
from arl.visibility.base import copy_visibility
from arl.visibility.operations import sum_visibility

log = logging.getLogger(__name__)


class TestImagingFunctions(unittest.TestCase):
    def setUp(self):
        import os
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
                       'wstack': 4.0}
    
    def actualSetUp(self, add_errors=False, freqwin=1, block=False, dospectral=True, dopol=False):
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
            flux = numpy.array([f* numpy.power(freq/1e8, -0.7) for freq in self.frequency])
        else:
            flux = numpy.array([f])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.componentvis = ingest_unittest_visibility(self.low, self.frequency, self.channelwidth, self.times,
                                                       self.vis_pol, self.phasecentre, block=block)
        
        self.model = create_unittest_model(self.componentvis, self.image_pol, npixel=self.params['npixel'])
        
        self.components = create_unittest_components(self.model, flux)
        self.model = insert_skycomponent(self.model, self.components)
        self.componentvis.data['vis'][...] = 0.0
        self.componentvis = predict_skycomponent_visibility(self.componentvis, self.components)
        
        self.cmodel = smooth_image(self.model)
        export_image_to_fits(self.model, '%s/test_imaging_functions_model.fits' % self.dir)
        export_image_to_fits(self.cmodel, '%s/test_imaging_functions_cmodel.fits' % self.dir)
        
        if add_errors:
            self.componentvis = insert_unittest_errors(self.componentvis)



    def _checkdirty(self, vis, context, fluxthreshold=0.3):
        # Make the dirty image
        self.params['imaginary'] = False
        self.params['timeslice'] = 'auto'
    
        dirty = create_empty_image_like(self.model)
        dirty, sumwt = invert_function(vis=vis, im=dirty, dopsf=False, normalize=True, context='2d', **self.params)
        export_image_to_fits(dirty, '%s/test_imaging_functions_%s_dirty.fits' % (self.dir, context))
        maxabs = numpy.max(numpy.abs(dirty.data))
        assert maxabs < fluxthreshold, "%s, abs max %f exceeds flux threshold" % (context, maxabs)

    def _checkcomponents(self, dirty, fluxthreshold=1.0, positionthreshold=1.0, check_dft=False):
        comps = find_skycomponents(dirty, fwhm=1.0, threshold=10 * fluxthreshold, npixels=5)
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

    def test_findcomponents(self):
        # Check that the components are where we expected them to be after insertion
        self.actualSetUp()
        self._checkcomponents(self.cmodel, check_dft=False)
    
    def test_findcomponents_spectral_pol(self):
        # Check that the components are where we expected them to be after insertion
        self.actualSetUp(dospectral=True, dopol=True)
        self._checkcomponents(self.cmodel, check_dft=False)
    
    def _predict_base(self, context='2d', extra='', fluxthreshold=1.0):
        self.modelvis = copy_visibility(self.componentvis, zero=True)
        self.modelvis.data['vis'] *= 0.0
        self.modelvis = predict_function(self.modelvis, self.model, context=context, **self.params)
        self.residualvis = copy_visibility(self.componentvis, zero=True)
        self.residualvis.data['uvw'][:, 2] = 0.0
        self.residualvis.data['vis'] = self.modelvis.data['vis'] - self.componentvis.data['vis']
        self._checkdirty(self.residualvis, 'predict_%s%s' % (context, extra),
                         fluxthreshold=fluxthreshold)
    
    def _invert_base(self, context, extra='', fluxthreshold=1.0, positionthreshold=1.0, check_components=True):
        dirty = create_empty_image_like(self.model)
        dirty, sumwt = invert_function(self.componentvis, dirty, dopsf=False, context=context, **self.params)
        export_image_to_fits(dirty, '%s/test_imaging_functions_invert_%s%s_dirty.fits' % (self.dir, context, extra))
        if check_components:
            self._checkcomponents(dirty, fluxthreshold, positionthreshold)
    
    def test_predict_2d(self):
        # Test if the 2D prediction works
        #
        # Set w=0 so that the two-dimensional transform should agree exactly with the component transform.
        # Good check on the grid correction in the image->vis direction
        # Set all w to zero
        self.actualSetUp()
        self.componentvis.data['uvw'][:, 2] = 0.0
        # Predict the visibility using direct evaluation
        self.componentvis.data['vis'][...]=0.0
        self.componentvis = predict_skycomponent_visibility(self.componentvis, self.components)
        
        self.modelvis = copy_visibility(self.componentvis, zero=True)
        self.modelvis.data['uvw'][:, 2] = 0.0
        self.modelvis = predict_2d(self.modelvis, self.model, **self.params)
        self.residualvis = copy_visibility(self.componentvis, zero=True)
        self.residualvis.data['uvw'][:, 2] = 0.0
        self.residualvis.data['vis'] = self.modelvis.data['vis'] - self.componentvis.data['vis']
        
        self._checkdirty(self.residualvis, 'predict_2d')
    
    def test_invert_2d(self):
        # Test if the 2D invert works with w set to zero
        # Set w=0 so that the two-dimensional transform should agree exactly with the model.
        # Good check on the grid correction in the vis->image direction
        
        self.actualSetUp()
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
    
    @unittest.skip("Intrinsically unstable")
    def test_predict_facets_wstack(self):
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['padding'] = 8
        self.actualSetUp()
        self._predict_base(context='facets_wstack')
    
    @unittest.skip("Interpolation insufficently accurate?")
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
        self._predict_base(context='wstack', extra='_spectral', fluxthreshold=7.0)
    
    def test_invert_facets(self):
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['padding'] = 8
        self.actualSetUp()
        self._invert_base(context='facets', positionthreshold=1.5, check_components=True)
    
    def test_invert_facets_timeslice(self):
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['padding'] = 8
        self.params['timeslice'] = None
        self.actualSetUp()
        self._invert_base(context='facets_timeslice', check_components=True,
                          positionthreshold=2.1)
    
    @unittest.skip("Seems to be correcting twice!")
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
        self.actualSetUp()
        self._invert_base(context='wstack', positionthreshold=1.0)
    
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
        im = create_image_from_visibility(self.componentvis, nchan=1, npixel=128)
        assert im.data.shape == (1, 1, 128, 128)
        im = create_image_from_visibility(self.componentvis, frequency=self.frequency, npixel=128)
        assert im.data.shape == (len(self.frequency), 1, 128, 128)
        im = create_image_from_visibility(self.componentvis, frequency=self.frequency, npixel=128,
                                          nchan=1)
        assert im.data.shape == (1, 1, 128, 128)


if __name__ == '__main__':
    unittest.main()
