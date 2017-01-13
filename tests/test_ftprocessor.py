"""Unit tests for Fourier transforms

realtimcornwell@gmail.com
"""
import logging
import os
import unittest

import numpy

from astropy.convolution import Gaussian2DKernel, convolve
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord
from numpy.testing import assert_allclose

from arl.fourier_transforms.ftprocessor import *
from arl.image.operations import export_image_to_fits
from arl.skymodel.operations import create_skycomponent, find_skycomponents, find_nearest_component, \
    insert_skycomponent
from arl.util.testing_support import create_named_configuration
from arl.visibility.operations import create_visibility, sum_visibility

log = logging.getLogger()
log.setLevel(logging.DEBUG)

log = logging.getLogger("tests.test_ftprocessor")


class TestFTProcessor(unittest.TestCase):
    def _checkdirty(self, vis, name='test_invert_2d_dirty', fluxthreshold=1.0):
        # Make the dirty image
        dirty = create_image_from_visibility(vis, **self.params)
        dirty, sumwt = invert_2d(vis=vis, im=dirty, dopsf=False, **self.params)
        dirty.data /= sumwt
        export_image_to_fits(dirty, '%s/%s_dirty.fits' % (self.dir, name))
        maxabs = numpy.max(numpy.abs(dirty.data))
        assert maxabs < fluxthreshold, "%s, abs max %f exceeds flux threshold" % (name, maxabs)
    
    def _checkcomponents(self, dirty, fluxthreshold=10.0, positionthreshold=1.0):
        comps = find_skycomponents(dirty, fwhm=1.0, threshold=10.0, npixels=5, params=None)
        assert len(comps) == len(self.components), "Different number of components found"
        cellsize = abs(dirty.wcs.wcs.cdelt[0])
        # Check for agreement between image and DFT - requires no visibility weighting
        for comp in comps:
            sflux = sum_visibility(self.componentvis, comp.direction, params=None)[0]
            assert abs(comp.flux[0, 0] - sflux[0,0]) < fluxthreshold, \
                "Fitted and DFT flux differ %s %s" % (comp.flux[0, 0], sflux[0, 0])
        # Check for agreement in direction
            ocomp = find_nearest_component(comp.direction, self.components)
            assert abs(comp.direction.ra.deg - ocomp.direction.ra.deg) / cellsize < \
               positionthreshold, \
            "Component differs in ra %s %s" % (comp.direction.ra.deg, ocomp.direction.ra.deg)
            assert abs(comp.direction.dec.deg - ocomp.direction.dec.deg) / cellsize < \
               positionthreshold, "Component differs in dec %s %s" % \
                                  (comp.direction.dec.deg, ocomp.direction.dec.deg)
    def setUp(self):
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        
        self.params = {'npixel': 512,
                       'npol': 1,
                       'cellsize': 0.0005,
                       'spectral_mode': 'channel',
                       'channelwidth': 5e7,
                       'reffrequency': 1e8,
                       'image_partitions': 8,
                       'padding': 2,
                       'oversampling': 8,
                       'timeslice':1.0,
                       'wstep': 2.0}
        
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = numpy.arange(- numpy.pi / 4.0, 1.001 * numpy.pi / 4.0, numpy.pi / 16.0)
        self.frequency = numpy.array([1e8])
        
        self.reffrequency = numpy.max(self.frequency)
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox=2000.0)
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                              phasecentre=self.phasecentre, **self.params)
        self.uvw = self.componentvis.data['uvw']
        self.flux = numpy.array([[100.0]])
        self.componentvis.data['vis'] *= 0.0
        
        self.model = create_image_from_visibility(self.componentvis, **self.params)
        self.model.data *= 0.0
        
        # Fill the visibility with exactly computed point sources. These are chosen to lie
        # on grid points.
        spacing_pixels = self.params['npixel'] // self.params['image_partitions']
        log.info('Spacing in pixels = %s' % spacing_pixels)

        centers = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]

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
                    flux = numpy.array([[100.0 + 2.0 * ix + iy * 20.0]])
                    comp = create_skycomponent(flux=flux, frequency=self.frequency, direction=sc)
                    self.components.append(comp)
                    insert_skycomponent(self.model, comp, **self.params)
            
        # Predict the visibility from the components exactly
        self.componentvis.data['vis'] *= 0.0
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)

        self.cmodel = create_image_from_array(convolve(self.model.data[0, 0, :, :], Gaussian2DKernel(3.0),
                                                  normalize_kernel=True), self.model.wcs)
        
        export_image_to_fits(self.model, '%s/test_model.fits' % self.dir)
        export_image_to_fits(self.cmodel, '%s/test_cmodel.fits' % self.dir)

    def test_predict_2d(self):
        """Test if the 2D prediction works

        Set w=0 so that the two-dimensional transform should agree exactly with the component transform.
        Good check on the grid correction in the image->vis direction"""
        # Set all w to zero
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                              phasecentre=self.phasecentre,
                                              **self.params)
        self.componentvis.data['uvw'][:, 2] = 0.0
        # Predict the visibility using direct evaluation
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)
        
        self.modelvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                          phasecentre=self.phasecentre, **self.params)
        self.modelvis.data['uvw'][:, 2] = 0.0
        predict_2d(self.modelvis, self.model, **self.params)
        self.residualvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                             phasecentre=self.phasecentre,
                                             **self.params)
        self.residualvis.data['uvw'][:, 2] = 0.0
        self.residualvis.data['vis'] = self.modelvis.data['vis'] - self.componentvis.data['vis']

        self._checkdirty(self.residualvis, 'test_predict_2d_residual', fluxthreshold=10.0)

    def _predict_base(self, predict, fluxthreshold=10.0):
        self.modelvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                          phasecentre=self.phasecentre, **self.params)
        self.modelvis.data['vis'] *= 0.0
        predict(self.modelvis, self.model, **self.params)
        self.residualvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                             phasecentre=self.phasecentre,
                                             **self.params)
        self.residualvis.data['uvw'][:, 2] = 0.0
        self.residualvis.data['vis'] = self.modelvis.data['vis'] - self.componentvis.data['vis']
        self._checkdirty(self.residualvis, 'test_%s_residual' % predict.__name__, fluxthreshold=fluxthreshold)


    def test_predict_by_image_partitions(self):
        self._predict_base(predict_by_image_partitions, fluxthreshold=1.0)
        
    def test_predict_timeslice(self):
        # This works very poorly because of the poor interpolation accuracy for point sources
        self.params['nprocessor']=1
        self._predict_base(predict_timeslice, fluxthreshold=20.0)

    def test_predict_timeslice_parallel(self):
        # This works very poorly because of the poor interpolation accuracy for point sources
        self.params['nprocessor']=4
        self._predict_base(predict_timeslice, fluxthreshold=20.0)

    def test_predict_wprojection(self):
        self.params = {'npixel': 512,
                       'npol': 1,
                       'cellsize': 0.0005,
                       'spectral_mode': 'channel',
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
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                              phasecentre=self.phasecentre,
                                              **self.params)
        self.componentvis.data['uvw'][:, 2] = 0.0
        self.componentvis.data['vis'] *= 0.0
        # Predict the visibility using direct evaluation
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)
    
        dirty2d = create_image_from_visibility(self.componentvis, **self.params)
        dirty2d, sumwt = invert_2d(self.componentvis, dirty2d, **self.params)

        dirty2d.data = dirty2d.data / sumwt
    
        export_image_to_fits(dirty2d, '%s/test_invert_2d_dirty.fits' % self.dir)

        self._checkcomponents(dirty2d, fluxthreshold=10.0, positionthreshold=1.0)

    def _invert_base(self, invert, fluxthreshold=10.0, positionthreshold=1.0):
        dirtyFacet = create_image_from_visibility(self.componentvis, **self.params)
        dirtyFacet, sumwt = invert(self.componentvis, dirtyFacet, **self.params)
        assert sumwt > 0.0
        dirtyFacet.data = dirtyFacet.data / sumwt
        export_image_to_fits(dirtyFacet, '%s/test_%s_dirty.fits' % (self.dir, invert.__name__))
        self._checkcomponents(dirtyFacet, fluxthreshold, positionthreshold)

    def test_invert_by_image_partitions(self):
        self._invert_base(invert_by_image_partitions, fluxthreshold=10.0, positionthreshold=1.0)

    def test_invert_timeslice(self):
        self.params['nprocessor']=1
        self._invert_base(invert_timeslice, fluxthreshold=10.0, positionthreshold=4.0)

    def test_invert_timeslice_parallel(self):
        self.params['nprocessor']=4
        self._invert_base(invert_timeslice, fluxthreshold=10.0, positionthreshold=4.0)

    def test_invert_wprojection(self):
        self.params = {'npixel': 512,
                       'npol': 1,
                       'cellsize': 0.0005,
                       'spectral_mode': 'channel',
                       'channelwidth': 5e7,
                       'reffrequency': 1e8,
                       'padding': 1,
                       'oversampling': 4,
                       'wstep': 2.0}
        self._invert_base(invert_wprojection, fluxthreshold=10.0, positionthreshold=1.0)


if __name__ == '__main__':
    import logging
    
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    unittest.main()
