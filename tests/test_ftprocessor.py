"""Unit tests for Fourier transforms

realtimcornwell@gmail.com
"""
import os
import unittest

from astropy.convolution import Gaussian2DKernel, convolve

from arl.fourier_transforms.ftprocessor import *
from arl.fourier_transforms.ftprocessor_timeslice import *
from arl.image.operations import export_image_to_fits, create_empty_image_like
from arl.skymodel.operations import create_skycomponent, find_skycomponents, find_nearest_component, \
    insert_skycomponent
from arl.util.testing_support import create_named_configuration, run_unittests
from arl.visibility.operations import create_visibility, sum_visibility

log = logging.getLogger(__name__)


class TestFTProcessor(unittest.TestCase):
    
    def _checkdirty(self, vis, name='test_invert_2d_dirty', fluxthreshold=1.0):
        # Make the dirty image
        dirty = create_empty_image_like(self.model)
        dirty, sumwt = invert_2d(vis=vis, im=dirty, dopsf=False, **self.params)
        dirty = normalize_sumwt(dirty, sumwt)
        export_image_to_fits(dirty, '%s/%s_dirty.fits' % (self.dir, name))
        maxabs = numpy.max(numpy.abs(dirty.data))
        assert maxabs < fluxthreshold, "%s, abs max %f exceeds flux threshold" % (name, maxabs)
    
    def _checkcomponents(self, dirty, fluxthreshold=10.0, positionthreshold=1.0):
        comps = find_skycomponents(dirty, fwhm=1.0, threshold=10.0, npixels=5)
        assert len(comps) == len(self.components), "Different number of components found"
        cellsize = abs(dirty.wcs.wcs.cdelt[0])
        # Check for agreement between image and DFT
        for comp in comps:
            sflux = sum_visibility(self.componentvis, comp.direction)[0]
            assert abs(comp.flux[0, 0] - sflux[0, 0]) < fluxthreshold, \
                "Fitted and DFT flux differ %s %s" % (comp.flux[0, 0], sflux[0, 0])
            # Check for agreement in direction
            ocomp = find_nearest_component(comp.direction, self.components)
            assert abs(comp.direction.ra.deg - ocomp.direction.ra.deg) / cellsize < \
                   positionthreshold, \
                "Component differs in dec %.3f pixels" % (comp.direction.ra.deg - ocomp.direction.ra.deg) / cellsize
            assert abs(comp.direction.dec.deg - ocomp.direction.dec.deg) / cellsize < \
                   positionthreshold, "Component differs in dec %.3f pixels" % \
                                      (comp.direction.dec.deg - ocomp.direction.dec.deg) / cellsize
    
    def setUp(self):
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        
        self.params = {'npixel': 512,
                       'npol': 1,
                       'image_nchan':1,
                       'cellsize': 0.0015,
                       'channelwidth': 5e6,
                       'reffrequency': 1e8,
                       'image_partitions': 8,
                       'padding': 2,
                       'oversampling': 8,
                       'timeslice': 'auto',
                       'wstep': 2.0}
        
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = numpy.arange(- 3.0, 3.1, 1.0)
        self.times = numpy.array([-0.0])
        log.info("Times are %s" % (self.times))
        self.times = (numpy.pi / 12.0) * self.times
        self.frequency = numpy.array([1e8])

        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox=2000.0)
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                              weight=1.0, npol=1)
        self.uvw = self.componentvis.data['uvw']
        self.componentvis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_image_from_visibility(self.componentvis, npixel=512, cellsize=0.0015,
                                                     image_nchan=1)

        # Fill the visibility with exactly computed point sources. These are chosen to lie
        # on grid points.
        spacing_pixels = 64
        log.info('Spacing in pixels = %s' % spacing_pixels)
        
        centers = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
        
        rpix = self.model.wcs.wcs.crpix - 1
        self.components = []
        for iy in centers:
            for ix in centers:
                if ix >= iy:
                    # The phase center in 0-relative coordinates is n // 2 so we centre the grid of
                    # components on ny // 2, nx // 2. The wcs must be defined consistently.
                    p = int(round(rpix[0] + ix * spacing_pixels * numpy.sign(self.model.wcs.wcs.cdelt[0]))), \
                        int(round(rpix[1] + iy * spacing_pixels * numpy.sign(self.model.wcs.wcs.cdelt[1])))
                    sc = pixel_to_skycoord(p[0], p[1], self.model.wcs)
                    log.info("Component at (%f, %f) [0-rel] %s" % (p[0], p[1], str(sc)))
                    
                    f = (100.0 + 1.0 * ix + iy * 10.0)
                    # Channel images
                    flux = numpy.array([[f]])
                    comp = create_skycomponent(flux=flux, frequency=[numpy.average(self.frequency)], direction=sc)
                    self.components.append(comp)
                    insert_skycomponent(self.model, comp)
       
        # Predict the visibility from the components exactly. We always do this for each spectral channel
        self.componentvis.data['vis'] *= 0.0
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)

        # Calculate the model convolved with a Gaussian.
        cmodel = create_image_from_array(convolve(self.model.data[0, 0, :, :], Gaussian2DKernel(1.5),
                                                       normalize_kernel=True), self.model.wcs)
        
        export_image_to_fits(self.model, '%s/test_model.fits' % self.dir)
        export_image_to_fits(cmodel, '%s/test_cmodel.fits' % self.dir)
    
    def test_predict_2d(self):
        """Test if the 2D prediction works

        Set w=0 so that the two-dimensional transform should agree exactly with the component transform.
        Good check on the grid correction in the image->vis direction"""
        # Set all w to zero
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                              weight=1.0, npol=1)
        self.componentvis.data['uvw'][:, 2] = 0.0
        # Predict the visibility using direct evaluation
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)
        
        self.modelvis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                          weight=1.0, npol=1)
        self.modelvis.data['uvw'][:, 2] = 0.0
        predict_2d(self.modelvis, self.model, **self.params)
        self.residualvis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                             weight=1.0, npol=1)
        self.residualvis.data['uvw'][:, 2] = 0.0
        self.residualvis.data['vis'] = self.modelvis.data['vis'] - self.componentvis.data['vis']
        
        self._checkdirty(self.residualvis, 'test_predict_2d_residual', fluxthreshold=10.0)
    
    def _predict_base(self, predict, fluxthreshold=10.0):
        self.modelvis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                          weight=1.0, npol=1)
        self.modelvis.data['vis'] *= 0.0
        predict(self.modelvis, self.model, **self.params)
        self.residualvis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                             weight=1.0, npol=1)
        self.residualvis.data['uvw'][:, 2] = 0.0
        self.residualvis.data['vis'] = self.modelvis.data['vis'] - self.componentvis.data['vis']
        self._checkdirty(self.residualvis, 'test_%s_residual' % predict.__name__, fluxthreshold=fluxthreshold)
    
    def test_predict_by_image_partitions(self):
        self._predict_base(predict_by_image_partitions, fluxthreshold=1.0)
    
    def test_predict_timeslice(self):
        # This works very poorly because of the poor interpolation accuracy for point sources
        self.params['nprocessor'] = 1
        self._predict_base(predict_timeslice, fluxthreshold=20.0)
    
    @unittest.skip("Do not run in VM")
    def test_predict_timeslice_parallel(self):
        # This works very poorly because of the poor interpolation accuracy for point sources
        self.params['nprocessor'] = 4
        self._predict_base(predict_timeslice, fluxthreshold=20.0)
    
    def test_predict_wprojection(self):
        self.params = {'npixel': 512,
                       'npol': 1,
                       'cellsize': 0.0015,
                       'channelwidth': 5e7,
                       'reffrequency': 1e8,
                       'padding': 1,
                       'oversampling': 8,
                       'wstep': 2.0}
        
        self._predict_base(predict_wprojection, fluxthreshold=4.0)
    
    def test_invert_2d(self):
        """Test if the 2D invert works with w set to zero

        Set w=0 so that the two-dimensional transform should agree exactly with the model.
        Good check on the grid correction in the vis->image direction
        """
        # Set all w to zero
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                              weight=1.0, npol=1)
        self.componentvis.data['uvw'][:, 2] = 0.0
        self.componentvis.data['vis'] *= 0.0
        # Predict the visibility using direct evaluation
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)
        
        dirty2d = create_empty_image_like(self.model)
        dirty2d, sumwt = invert_2d(self.componentvis, dirty2d, **self.params)
        
        dirty2d = normalize_sumwt(dirty2d, sumwt)
        
        export_image_to_fits(dirty2d, '%s/test_invert_2d_dirty.fits' % self.dir)
        
        self._checkcomponents(dirty2d, fluxthreshold=10.0, positionthreshold=1.0)
    
    def _invert_base(self, invert, fluxthreshold=10.0, positionthreshold=1.0):
        dirtyFacet = create_empty_image_like(self.model)
        dirtyFacet, sumwt = invert(self.componentvis, dirtyFacet, **self.params)
        assert sumwt.all() > 0.0
        dirtyFacet = normalize_sumwt(dirtyFacet, sumwt)
        export_image_to_fits(dirtyFacet, '%s/test_%s_dirty.fits' % (self.dir, invert.__name__))
        self._checkcomponents(dirtyFacet, fluxthreshold, positionthreshold)
    
    def test_invert_by_image_partitions(self):
        self._invert_base(invert_by_image_partitions, fluxthreshold=10.0, positionthreshold=1.0)
    
    def test_invert_timeslice(self):
        self._invert_base(invert_timeslice, fluxthreshold=10.0, positionthreshold=8.0)
    
    @unittest.skip("Do not run in VM")
    def test_invert_timeslice_parallel(self):
        self.params['nprocessor'] = 4
        self._invert_base(invert_timeslice, fluxthreshold=10.0, positionthreshold=4.0)
    
    def test_invert_wprojection(self):
        self.params = {'npixel': 512,
                       'npol': 1,
                       'cellsize': 0.0015,
                       'channelwidth': 5e7,
                       'reffrequency': 1e8,
                       'padding': 1,
                       'oversampling': 4,
                       'wstep': 2.0}
        self._invert_base(invert_wprojection, fluxthreshold=10.0, positionthreshold=1.0)


if __name__ == '__main__':
    run_unittests()
