"""Unit tests for image operations

realtimcornwell@gmail.com
"""
import logging
import os
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from arl.data.polarisation import PolarisationFrame
from arl.fourier_transforms.ftprocessor_base import predict_2d, invert_2d, create_image_from_visibility, normalize_sumwt
from arl.image.deconvolution import overlapIndices, deconvolve_cube, restore_cube
from arl.image.operations import export_image_to_fits, create_image_from_array
from arl.util.testing_support import create_test_image, create_named_configuration
from arl.visibility.operations import create_visibility

log = logging.getLogger(__name__)


class TestImageDeconvolution(unittest.TestCase):
    def setUp(self):
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / (12.0)) * numpy.linspace(-3.0, 3.0, 7)
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e6])
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox=2000.0)
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('stokesI'))
        self.vis.data['vis'] *= 0.0
        
        # Create model
        self.test_model = create_test_image(cellsize=0.001, phasecentre=self.vis.phasecentre,
                                            frequency=self.frequency)
        self.vis = predict_2d(self.vis, self.test_model)
        assert numpy.max(numpy.abs(self.vis.vis)) > 0.0
        self.model = create_image_from_visibility(self.vis, npixel=512, cellsize=0.001,
                                                  polarisation_frame=PolarisationFrame('stokesI'))
        self.dirty, sumwt = invert_2d(self.vis, self.model)
        self.dirty = normalize_sumwt(self.dirty, sumwt)
        self.psf, sumwt = invert_2d(self.vis, self.model, dopsf=True)
        self.psf = normalize_sumwt(self.psf, sumwt)
        window = numpy.zeros(shape=self.model.shape, dtype=numpy.bool)
        window[..., 129:384, 129:384] = True
        self.innerquarter = create_image_from_array(window, self.model.wcs)
    
    def overlaptest(self, a1, a2, s1, s2):
        #
        a1[s1[0]:s1[1], s1[2]:s1[3]] = 1
        a2[s2[0]:s2[1], s2[2]:s2[3]] = 1
        return numpy.sum(a1) == numpy.sum(a2)
    
    def test_overlapindices_same(self):
        a1 = numpy.zeros([256, 256], dtype='int')
        a2 = numpy.zeros([256, 256], dtype='int')
        shiftx = 20
        shifty = -12
        s1, s2 = overlapIndices(a1, a2, shiftx, shifty)
        assert len(s1) == 4
        assert len(s2) == 4
        assert s1 == (20, 256, 0, 244)
        assert s2 == (0, 236, 12, 256)
        self.overlaptest(a1, a2, s1, s2)
    
    def test_overlapindices_differ(self):
        a1 = numpy.zeros([256, 256], dtype='int')
        a2 = numpy.zeros([32, 32], dtype='int')
        shiftx = 20
        shifty = -12
        s1, s2 = overlapIndices(a1, a2, shiftx, shifty)
        assert len(s1) == 4
        assert len(s2) == 4
        assert s1 == (20, 256, 0, 244)
        assert s2 == (0, 12, 12, 32)
        self.overlaptest(a1, a2, s1, s2)
    
    def test_deconvolve_hogbom(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=10000, gain=0.1, algorithm='hogbom',
                                                   threshold=0.01)
        export_image_to_fits(self.residual, "%s/test_deconvolve_hogbom-residual.fits" % (self.dir))
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        export_image_to_fits(self.cmodel, "%s/test_deconvolve_hogbom-clean.fits" % (self.dir))
        assert numpy.max(self.residual.data) < 0.25
    
    def test_deconvolve_msclean(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=1000,
                                                   gain=0.7, algorithm='msclean', scales=[0, 30, 10, 30],
                                                   threshold=0.01)
        export_image_to_fits(self.residual, "%s/test_deconvolve_msclean-residual.fits" % (self.dir))
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        export_image_to_fits(self.cmodel, "%s/test_deconvolve_msclean-clean.fits" % (self.dir))
        assert numpy.max(self.residual.data) < 0.7
    
    def test_deconvolve_hogbom_inner_quarter(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, window=self.innerquarter, niter=10000,
                                                   gain=0.1, algorithm='hogbom', threshold=0.01)
        export_image_to_fits(self.residual, "%s/test_deconvolve_hogbom_innerquarter-residual.fits" % (self.dir))
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        export_image_to_fits(self.cmodel, "%s/test_deconvolve_hogbom_innerquarter-clean.fits" % (self.dir))
        assert numpy.max(self.residual.data) < 0.25
    
    def test_deconvolve_msclean_inner_quarter(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, window=self.innerquarter, niter=1000, gain=0.7,
                                                   algorithm='msclean', scales=[0, 30, 10, 30], threshold=0.01)
        export_image_to_fits(self.residual, "%s/test_deconvolve_msclean_innerquarter-residual.fits" % (self.dir))
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        export_image_to_fits(self.cmodel, "%s/test_deconvolve_msclean_innerquarter-clean.fits" % (self.dir))
        assert numpy.max(self.residual.data) < 0.5
