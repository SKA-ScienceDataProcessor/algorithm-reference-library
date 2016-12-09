"""Unit tests for Fourier transforms

realtimcornwell@gmail.com
"""
import logging
import os
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord
from numpy.testing import assert_allclose

from arl.fourier_transforms.ftprocessor import invert_2d, predict_2d, create_image_from_visibility, \
    predict_skycomponent_visibility, invert_by_image_partitions, \
    predict_by_image_partitions
from arl.image.operations import export_image_to_fits
from arl.skymodel.operations import create_skycomponent, insert_skycomponent
from arl.util.testing_support import create_named_configuration
from arl.visibility.operations import create_visibility

log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

log = logging.getLogger("tests.test_ftprocessor")

class TestFTProcessor(unittest.TestCase):
    
    def _checkdirty(self, vis, name='test_invert_2d_dirty', invert=invert_2d):
        # Make the dirty image and PSF
        dirty = create_image_from_visibility(vis, params=self.params)
        dirty = invert(vis=vis, im=dirty, dopsf=False, params=self.params)
        psf = create_image_from_visibility(vis, params=self.params)
        psf = invert(vis=vis, im=psf, dopsf=True, params=self.params)
        psfmax = psf.data.max()
        dirty.data /= psfmax
        psf.data /= psfmax
        export_image_to_fits(dirty, '%s/%s_%s_dirty.fits' % (self.dir, name, self.params['kernel']))
        export_image_to_fits(psf,   '%s/%s_%s_psf.fits' % (self.dir, name, self.params['kernel']))

    def setUp(self):
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        
        self.params = {'npixel': 512,
                       'cellsize': 0.001,
                       'spectral_mode': 'channel',
                       'channelwidth': 5e7,
                       'reffrequency': 1e8,
                       'image_partitions': 5,
                       'padding':1,
                       'kernel':'transform',
                       'oversampling':8}
        
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = numpy.arange(-numpy.pi / 2.0, +numpy.pi / 2.0, 0.05)
        self.frequency = numpy.array([1e8])
        
        self.reffrequency = numpy.max(self.frequency)
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                              phasecentre=self.phasecentre)
        self.uvw = self.componentvis.data['uvw']
        self.flux = numpy.array([[100.0, 80, -20.0, 1.0]])
        self.componentvis.data['vis'] *= 0.0
        
        self.model = create_image_from_visibility(self.componentvis, params=self.params)
        self.model.data *= 0.0

        # Fill the visibility with exactly computed point sources. These are chosen to lie
        # on grid points.
        spacing_pixels = self.params['npixel'] // self.params['image_partitions']
        log.info('Spacing in pixels = %s' % spacing_pixels)
        centers = [-2.0, -1.0, 0.0, 1.0, 2.0]
        self.components = []
        for iy in centers:
            for ix in centers:
                pra, pdec = self.params['npixel'] // 2 + ix * spacing_pixels, \
                            self.params['npixel'] // 2 + iy * spacing_pixels
                sc = pixel_to_skycoord(pra, pdec, self.model.wcs)
                log.info("Component at (%f, %f) %s" % (pra, pdec, str(sc)))
                comp = create_skycomponent(flux=self.flux, frequency=self.frequency, direction=sc)
                self.components.append(comp)
                self.model.data[..., int(pra), int(pdec)] += self.flux

        # Predict the visibility from the components exactly
        self.componentvis.data['vis'] *= 0.0
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)
        
        export_image_to_fits(self.model, '%s/test_model.fits' % self.dir)

    def test_predict_wprojection(self):
        """Test if the w projection works

        Set w=0 so that the two-dimensional transform should agree exactly with the component transform.
        Good check on the grid correction in the image->vis direction"""
        self.modelvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                          phasecentre=self.phasecentre)
        self.modelvis.data['uvw'][:, 2] = 0.0
        self.params['kernel']='wprojection'
        predict_2d(self.modelvis, self.model, params=self.params)
        self.residualvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                             phasecentre=self.phasecentre,
                                             params=self.params)
        self.residualvis.data['uvw'][:, 2] = 0.0
        self.residualvis.data['vis'] = self.modelvis.data['vis'] - self.componentvis.data['vis']
        self._checkdirty(self.residualvis, 'test_predict_wprojection_residual')
    
        assert_allclose(self.modelvis.data['vis'].real, self.componentvis.data['vis'].real, atol=1.0)
        assert_allclose(self.modelvis.data['vis'].imag, self.componentvis.data['vis'].imag, atol=1.0)

    def test_predict_2d(self):
        """Test if the 2D prediction works

        Set w=0 so that the two-dimensional transform should agree exactly with the component transform.
        Good check on the grid correction in the image->vis direction"""
        # Set all w to zero
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                              phasecentre=self.phasecentre,
                                              params=self.params)
        self.componentvis.data['uvw'][:, 2] = 0.0
        # Predict the visibility using direct evaluation
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)
    
        self.modelvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                          phasecentre=self.phasecentre)
        self.modelvis.data['uvw'][:, 2] = 0.0
        predict_2d(self.modelvis, self.model, params=self.params)
        self.residualvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                             phasecentre=self.phasecentre,
                                             params=self.params)
        self.residualvis.data['uvw'][:, 2] = 0.0
        self.residualvis.data['vis'] = self.modelvis.data['vis'] - self.componentvis.data['vis']
        self._checkdirty(self.residualvis, 'test_predict_2d_residual')
    
        assert_allclose(self.modelvis.data['vis'].real, self.componentvis.data['vis'].real, atol=1.0)
        assert_allclose(self.modelvis.data['vis'].imag, self.componentvis.data['vis'].imag, atol=1.0)

    def test_predict_2d_byrows(self):
        """Test if the 2D prediction works

        Set w=0 so that the two-dimensional transform should agree exactly with the component transform.
        Good check on the grid correction in the image->vis direction"""
        # Set all w to zero
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                              phasecentre=self.phasecentre,
                                              params=self.params)
        self.componentvis.data['uvw'][:, 2] = 0.0
        # Predict the visibility using direct evaluation
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)
    
        self.modelvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                          phasecentre=self.phasecentre)
        self.modelvis.data['uvw'][:, 2] = 0.0
        self.params['kernel']='standard-by-row'
        predict_2d(self.modelvis, self.model, params=self.params)
        self.residualvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                             phasecentre=self.phasecentre,
                                             params=self.params)
        self.residualvis.data['uvw'][:, 2] = 0.0
        self.residualvis.data['vis'] = self.modelvis.data['vis'] - self.componentvis.data['vis']
        self._checkdirty(self.residualvis, 'test_predict_2d_residual')
    
        assert_allclose(self.modelvis.data['vis'].real, self.componentvis.data['vis'].real, atol=1.0)
        assert_allclose(self.modelvis.data['vis'].imag, self.componentvis.data['vis'].imag, atol=1.0)

    def test_invert_2d(self):
        """Test if the 2D invert works

        Set w=0 so that the two-dimensional transform should agree exactly with the model.
        Good check on the grid correction in the vis->image direction
        """
        # Set all w to zero
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                              phasecentre=self.phasecentre,
                                              params=self.params)
        self.componentvis.data['uvw'][:, 2] = 0.0
        self.componentvis.data['vis'] *= 0.0
        # Predict the visibility using direct evaluation
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)
        
        for self.params['kernel'] in ['calculate']:
            self._checkdirty(self.componentvis, 'test_invert_2d')
    
    def test_invert_image_partition(self):
        """Test if the image partition invert works

        If the faceting is fine enough, the dirty image should agree with the model.
        """
        dirtyFacet = create_image_from_visibility(self.componentvis, params=self.params)
        dirtyFacet = invert_by_image_partitions(self.componentvis, dirtyFacet, params=self.params)
        psfFacet = create_image_from_visibility(self.componentvis, params=self.params)
        psfFacet = invert_by_image_partitions(vis=self.componentvis, im=psfFacet, dopsf=True,
                                          params=self.params)
        psfmax = psfFacet.data.max()
        assert psfmax > 0.0
        dirtyFacet.data = dirtyFacet.data / psfmax
        
        export_image_to_fits(dirtyFacet, '%s/test_invert_image_partition_dirty.fits' % self.dir)
        export_image_to_fits(psfFacet, '%s/test_invert_image_partition_psf.fits' % self.dir)
        
    def test_predict_image_partition(self):
        """Test if the image partition predict works

        If the faceting is fine enough, the visibility should agree with the model visibility
        """
        self.modelvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                          phasecentre=self.phasecentre)
        self.modelvis.data['vis'] *= 0.0
        predict_by_image_partitions(self.modelvis, self.model, params=self.params)
        
        self.residualvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                              phasecentre=self.phasecentre,
                                              params=self.params)
        self.residualvis.data['uvw'][:, 2] = 0.0
        self.residualvis.data['vis'] = self.modelvis.data['vis']- self.componentvis.data['vis']
        self._checkdirty(self.residualvis, 'test_predict_image_partition_residual')

        assert_allclose(self.modelvis.data['vis'].real, self.componentvis.data['vis'].real, atol=1.0)
        assert_allclose(self.modelvis.data['vis'].imag, self.componentvis.data['vis'].imag, atol=1.0)


if __name__ == '__main__':
    import sys
    import logging
    
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    unittest.main()
