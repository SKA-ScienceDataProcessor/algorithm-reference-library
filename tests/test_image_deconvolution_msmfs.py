"""Unit tests for image deconvolution vis MSMFS


"""
import logging
import os
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from arl.data.polarisation import PolarisationFrame
from arl.image.deconvolution import deconvolve_cube, restore_cube
from arl.image.operations import export_image_to_fits, create_image_from_array
from arl.util.testing_support import create_low_test_image_from_gleam, create_low_test_beam, create_named_configuration
from arl.visibility.base import create_visibility
from arl.imaging.base import predict_2d, invert_2d, create_image_from_visibility

log = logging.getLogger(__name__)


class TestImageDeconvolutionMSMFS(unittest.TestCase):
    def setUp(self):
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        self.niter = 1000
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.nchan = 5
        self.times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 7)
        self.frequency = numpy.linspace(0.9e8, 1.1e8, self.nchan)
        self.channel_bandwidth = numpy.array(self.nchan * [self.frequency[1] - self.frequency[0]])
        self.phasecentre = SkyCoord(ra=+0.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('stokesI'))
        self.vis.data['vis'] *= 0.0
        
        # Create model
        self.test_model = create_low_test_image_from_gleam(npixel=512, cellsize=0.001,
                                                           phasecentre=self.vis.phasecentre,
                                                           frequency=self.frequency,
                                                           channel_bandwidth=self.channel_bandwidth)
        beam = create_low_test_beam(self.test_model)
        export_image_to_fits(beam, "%s/test_deconvolve_msmfsclean_beam.fits" % self.dir)
        self.test_model.data *= beam.data
        export_image_to_fits(self.test_model, "%s/test_deconvolve_msmfsclean_model.fits" % self.dir)
        self.vis = predict_2d(self.vis, self.test_model)
        assert numpy.max(numpy.abs(self.vis.vis)) > 0.0
        self.model = create_image_from_visibility(self.vis, npixel=512, cellsize=0.001,
                                                  polarisation_frame=PolarisationFrame('stokesI'))
        self.dirty, sumwt = invert_2d(self.vis, self.model)
        self.psf, sumwt = invert_2d(self.vis, self.model, dopsf=True)
        export_image_to_fits(self.dirty, "%s/test_deconvolve_msmfsclean_dirty.fits" % self.dir)
        export_image_to_fits(self.psf, "%s/test_deconvolve_msmfsclean_psf.fits" % self.dir)
        window = numpy.ones(shape=self.model.shape, dtype=numpy.bool)
        window[..., 129:384, 129:384] = True
        self.innerquarter = create_image_from_array(window, self.model.wcs)
    
    def test_deconvolve_msmfsclean_no_taylor(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=self.niter, gain=0.1,
                                                   algorithm='msmfsclean',
                                                   scales=[0, 3, 10], threshold=0.01, nmoments=1, findpeak='ARL',
                                                   fractional_threshold=0.01, window=self.innerquarter)
        export_image_to_fits(self.comp, "%s/test_deconvolve_msmfsclean_notaylor-comp.fits" % self.dir)
        export_image_to_fits(self.residual, "%s/test_deconvolve_msmfsclean_notaylor-residual.fits" % self.dir)
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        export_image_to_fits(self.cmodel, "%s/test_deconvolve_msmfsclean_notaylor-clean.fits" % self.dir)
        assert numpy.max(self.residual.data) < 1.4
    
    def test_deconvolve_msmfsclean_no_taylor_noscales(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=self.niter, gain=0.1,
                                                   algorithm='msmfsclean',
                                                   scales=[0], threshold=0.01, nmoments=1, findpeak='ARL',
                                                   fractional_threshold=0.01, window=self.innerquarter)
        export_image_to_fits(self.comp, "%s/test_deconvolve_msmfsclean_notaylor_noscales-comp.fits" % self.dir)
        export_image_to_fits(self.residual, "%s/test_deconvolve_msmfsclean_notaylor_noscales-residual.fits" % self.dir)
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        export_image_to_fits(self.cmodel, "%s/test_deconvolve_msmfsclean_notaylor_noscales-clean.fits" % self.dir)
        assert numpy.max(self.residual.data) < 1.4
    
    def test_deconvolve_msmfsclean_linear(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=self.niter, gain=0.1,
                                                   algorithm='msmfsclean',
                                                   scales=[0, 3, 10], threshold=0.01, nmoments=2, findpeak='ARL',
                                                   fractional_threshold=0.01, window=self.innerquarter)
        export_image_to_fits(self.comp, "%s/test_deconvolve_msmfsclean_linear-comp.fits" % self.dir)
        export_image_to_fits(self.residual, "%s/test_deconvolve_msmfsclean_linear-residual.fits" % self.dir)
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        export_image_to_fits(self.cmodel, "%s/test_deconvolve_msmfsclean_linear-clean.fits" % self.dir)
        assert numpy.max(self.residual.data) < 1.4
    
    def test_deconvolve_msmfsclean_linear_noscales(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=self.niter, gain=0.1,
                                                   algorithm='msmfsclean',
                                                   scales=[0], threshold=0.01, nmoments=2, findpeak='ARL',
                                                   fractional_threshold=0.01, window=self.innerquarter)
        export_image_to_fits(self.comp, "%s/test_deconvolve_msmfsclean_linear_noscales-comp.fits" % self.dir)
        export_image_to_fits(self.residual, "%s/test_deconvolve_msmfsclean_linear_noscales-residual.fits" % self.dir)
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        export_image_to_fits(self.cmodel, "%s/test_deconvolve_msmfsclean_linear_noscales-clean.fits" % self.dir)
        assert numpy.max(self.residual.data) < 1.4
    
    def test_deconvolve_msmfsclean_quadratic(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=self.niter, gain=0.1,
                                                   algorithm='msmfsclean',
                                                   scales=[0, 3, 10], threshold=0.01, nmoments=2, findpeak='ARL',
                                                   fractional_threshold=0.01, window=self.innerquarter)
        export_image_to_fits(self.comp, "%s/test_deconvolve_msmfsclean_quadratic-comp.fits" % self.dir)
        export_image_to_fits(self.residual, "%s/test_deconvolve_msmfsclean_quadratic-residual.fits" % self.dir)
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        export_image_to_fits(self.cmodel, "%s/test_deconvolve_msmfsclean_quadratic-clean.fits" % self.dir)
        assert numpy.max(self.residual.data) < 1.4
    
    def test_deconvolve_msmfsclean_quadratic_noscales(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=self.niter, gain=0.1,
                                                   algorithm='msmfsclean',
                                                   scales=[0], threshold=0.01, nmoments=2, findpeak='ARL',
                                                   fractional_threshold=0.01, window=self.innerquarter)
        export_image_to_fits(self.comp, "%s/test_deconvolve_msmfsclean_quadratic_noscales-comp.fits" % self.dir)
        export_image_to_fits(self.residual, "%s/test_deconvolve_msmfsclean_quadratic_noscales-residual.fits" % self.dir)
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        export_image_to_fits(self.cmodel, "%s/test_deconvolve_msmfsclean_quadratic_noscales-clean.fits" % self.dir)
        assert numpy.max(self.residual.data) < 1.4

    def test_deconvolve_msmfsclean_quadratic_psf(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=self.niter, gain=0.1,
                                                   algorithm='msmfsclean',
                                                   scales=[0, 3, 10], threshold=0.01, nmoments=2, findpeak='ARL',
                                                   fractional_threshold=0.01, window=self.innerquarter,
                                                   psf_support=32)
        export_image_to_fits(self.comp, "%s/test_deconvolve_msmfsclean_quadratic_psf-comp.fits" % self.dir)
        export_image_to_fits(self.residual, "%s/test_deconvolve_msmfsclean_quadratic_psf-residual.fits" % self.dir)
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        export_image_to_fits(self.cmodel, "%s/test_deconvolve_msmfsclean_quadratic_psf-clean.fits" % self.dir)
        assert numpy.max(self.residual.data) < 1.4


if __name__ == '__main__':
    unittest.main()
