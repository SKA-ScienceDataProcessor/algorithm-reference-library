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
    predict_skycomponent_visibility, invert_image_partition, \
    predict_image_partition
from arl.image.operations import export_image_to_fits
from arl.skymodel.operations import create_skycomponent, insert_skycomponent
from arl.util.testing_support import create_named_configuration
from arl.visibility.operations import create_visibility

log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

log = logging.getLogger("tests.test_ftprocessor")


class TestFTProcessor(unittest.TestCase):
    def setUp(self):
        self.dir = './test_ftprocessor_results'
        os.makedirs(self.dir, exist_ok=True)
        
        self.params = {'npixel': 512,
                       'cellsize': 0.001,
                       'spectral_mode': 'channel',
                       'channelwidth': 5e7,
                       'reffrequency': 1e8,
                       'image_partitions': 5}
        
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

        # Fill the visibility with exactly computed point sources
        spacing_pixels = self.params['npixel'] // self.params['image_partitions']
        log.info('Spacing in pixels = %s' % spacing_pixels)
        spacing = 180.0 * self.params['cellsize'] * spacing_pixels / numpy.pi
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

        self.componentvis.data['vis'] *= 0.0
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)
        
        export_image_to_fits(self.model, '%s/test_insert_component.fits' % self.dir)
    
    def test_roundtrip_2d(self):
        # Predict the visibility using direct evaluation with zero w
        self.componentvis.data['uvw'][:, 2] = 0.0
        self.componentvis.data['vis'] *= 0.0
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)
            insert_skycomponent(self.model, comp)
        
        self.componentvis = predict_2d(model=self.model, vis=self.componentvis, kernel=None, params=self.params)
        # Make images
        self.dirty = invert_2d(vis=self.componentvis, im=self.model, dopsf=False, kernel=None, params=self.params)
        self.psf = invert_2d(vis=self.componentvis, im=self.model, dopsf=True, kernel=None, params=self.params)
        psfmax = self.psf.data.max()
        self.dirty.data /= psfmax
        self.psf.data /= psfmax
        
        export_image_to_fits(self.dirty, '%s/test_roundtrip_dirty.fits' % self.dir)
        export_image_to_fits(self.psf, '%s/test_roundtrip_psf.fits' % self.dir)
    
    def test_predict_2d(self):
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
        self.residualvis.data['vis'] = self.modelvis.data['vis']- self.componentvis.data['vis']
        self.dirty = invert_2d(vis=self.residualvis, im=self.model, dopsf=False, kernel=None, params=self.params)
        export_image_to_fits(self.dirty, '%s/test_predict_2d_residual.fits' % self.dir)

        assert_allclose(self.modelvis.data['vis'].real, self.componentvis.data['vis'].real, atol=1.0)
        assert_allclose(self.modelvis.data['vis'].imag, self.componentvis.data['vis'].imag, atol=1.0)

    def test_invert_2d(self):
        # Set all w to zero
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                              phasecentre=self.phasecentre,
                                              params=self.params)
        self.componentvis.data['uvw'][:, 2] = 0.0
        # Predict the visibility using direct evaluation
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)
        
        self.dirty = create_image_from_visibility(self.componentvis, params=self.params)
        self.dirty = invert_2d(vis=self.componentvis, im=self.model, dopsf=False, kernel=None, params=self.params)
        self.psf = create_image_from_visibility(self.componentvis, params=self.params)
        self.psf = invert_2d(vis=self.componentvis, im=self.model, dopsf=True, kernel=None, params=self.params)
        psfmax = self.psf.data.max()
        self.dirty.data /= psfmax
        self.psf.data /= psfmax
        export_image_to_fits(self.dirty, '%s/test_invert_2d_dirty.fits' % self.dir)
        export_image_to_fits(self.psf, '%s/test_invert_2d_psf.fits' % self.dir)
    
    def test_invert_image_partition(self):
        dirtyFacet = create_image_from_visibility(self.componentvis, params=self.params)
        dirtyFacet = invert_image_partition(self.componentvis, dirtyFacet, params=self.params)
        psfFacet = create_image_from_visibility(self.componentvis, params=self.params)
        psfFacet = invert_image_partition(vis=self.componentvis, im=psfFacet, dopsf=True, kernel=None,
                                          params=self.params)
        psfmax = psfFacet.data.max()
        assert psfmax > 0.0
        dirtyFacet.data = dirtyFacet.data / psfmax
        
        export_image_to_fits(dirtyFacet, '%s/test_image_partition_tdirty.fits' % self.dir)
        export_image_to_fits(psfFacet, '%s/test_image_partition_psf.fits' % self.dir)
        
    def test_predict_image_partition(self):
        self.modelvis = create_visibility(self.lowcore, self.times, self.frequency, weight=1.0,
                                          phasecentre=self.phasecentre)
        predict_image_partition(self.modelvis, self.model, params=self.params)
        assert_allclose(self.modelvis.data['vis'].real, self.componentvis.data['vis'].real, atol=1.0)
        assert_allclose(self.modelvis.data['vis'].imag, self.componentvis.data['vis'].imag, atol=1.0)


if __name__ == '__main__':
    import sys
    import logging
    
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    unittest.main()
