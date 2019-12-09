"""Unit tests for image deconvolution vis MSMFS


"""
import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame

from processing_library.image.operations import create_image_from_array

from processing_components.image.deconvolution import deconvolve_cube, restore_cube
from processing_components.image.operations import export_image_to_fits
from processing_components.simulation import create_low_test_image_from_gleam
from processing_components.simulation import create_named_configuration
from processing_components.imaging.primary_beams import create_low_test_beam
from processing_components.visibility.base import create_visibility
from processing_components.imaging.base import predict_2d, invert_2d, create_image_from_visibility

log = logging.getLogger(__name__)


class TestImageDeconvolutionMSMFS(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        self.persist = False
        self.niter = 1000
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.nchan = 5
        self.times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 7)
        self.frequency = numpy.linspace(0.9e8, 1.1e8, self.nchan)
        self.channel_bandwidth = numpy.array(self.nchan * [self.frequency[1] - self.frequency[0]])
        self.phasecentre = SkyCoord(ra=+0.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('stokesI'), zerow=True)
        self.vis.data['vis'] *= 0.0
        
        # Create model
        self.test_model = create_low_test_image_from_gleam(npixel=512, cellsize=0.001,
                                                           phasecentre=self.vis.phasecentre,
                                                           frequency=self.frequency,
                                                           channel_bandwidth=self.channel_bandwidth,
                                                           flux_limit=1.0)
        beam = create_low_test_beam(self.test_model)
        if self.persist: export_image_to_fits(beam, "%s/test_deconvolve_mmclean_beam.fits" % self.dir)
        self.test_model.data *= beam.data
        if self.persist: export_image_to_fits(self.test_model, "%s/test_deconvolve_mmclean_model.fits" % self.dir)
        self.vis = predict_2d(self.vis, self.test_model)
        assert numpy.max(numpy.abs(self.vis.vis)) > 0.0
        self.model = create_image_from_visibility(self.vis, npixel=512, cellsize=0.001,
                                                  polarisation_frame=PolarisationFrame('stokesI'))
        self.dirty, sumwt = invert_2d(self.vis, self.model)
        self.psf, sumwt = invert_2d(self.vis, self.model, dopsf=True)
        if self.persist: export_image_to_fits(self.dirty, "%s/test_deconvolve_mmclean-dirty.fits" % self.dir)
        if self.persist: export_image_to_fits(self.psf, "%s/test_deconvolve_mmclean-psf.fits" % self.dir)
        window = numpy.ones(shape=self.model.shape, dtype=numpy.bool)
        window[..., 129:384, 129:384] = True
        self.innerquarter = create_image_from_array(window, self.model.wcs, polarisation_frame=PolarisationFrame('stokesI'))
    
    def test_deconvolve_mmclean_no_taylor(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=self.niter, gain=0.1,
                                                   algorithm='mmclean',
                                                   scales=[0, 3, 10], threshold=0.01, nmoment=1, findpeak='ARL',
                                                   fractional_threshold=0.01, window=self.innerquarter)
        if self.persist: export_image_to_fits(self.comp, "%s/test_deconvolve_mmclean_notaylor-comp.fits" % self.dir)
        if self.persist: export_image_to_fits(self.residual, "%s/test_deconvolve_mmclean_notaylor-residual.fits" % self.dir)
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist: export_image_to_fits(self.cmodel, "%s/test_deconvolve_mmclean_notaylor-clean.fits" % self.dir)
        assert numpy.max(self.residual.data) < 3.0

    def test_deconvolve_mmclean_no_taylor_edge(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=self.niter, gain=0.1,
                                                   algorithm='mmclean',
                                                   scales=[0, 3, 10], threshold=0.01, nmoment=1, findpeak='ARL',
                                                   fractional_threshold=0.01, window_shape='no_edge', window_edge=32)
        if self.persist: export_image_to_fits(self.comp, "%s/test_deconvolve_mmclean_notaylor-comp.fits" % self.dir)
        if self.persist: export_image_to_fits(self.residual, "%s/test_deconvolve_mmclean_notaylor-residual.fits" % self.dir)
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist: export_image_to_fits(self.cmodel, "%s/test_deconvolve_mmclean_notaylor-clean.fits" % self.dir)
        assert numpy.max(self.residual.data) < 3.0

    def test_deconvolve_mmclean_no_taylor_noscales(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=self.niter, gain=0.1,
                                                   algorithm='mmclean',
                                                   scales=[0], threshold=0.01, nmoment=1, findpeak='ARL',
                                                   fractional_threshold=0.01, window=self.innerquarter)
        if self.persist: export_image_to_fits(self.comp, "%s/test_deconvolve_mmclean_notaylor_noscales-comp.fits" % self.dir)
        if self.persist: export_image_to_fits(self.residual, "%s/test_deconvolve_mmclean_notaylor_noscales-residual.fits" % self.dir)
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist: export_image_to_fits(self.cmodel, "%s/test_deconvolve_mmclean_notaylor_noscales-clean.fits" % self.dir)
        assert numpy.max(self.residual.data) < 3.0
    
    def test_deconvolve_mmclean_linear(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=self.niter, gain=0.1,
                                                   algorithm='mmclean',
                                                   scales=[0, 3, 10], threshold=0.01, nmoment=2, findpeak='ARL',
                                                   fractional_threshold=0.01, window=self.innerquarter)
        if self.persist: export_image_to_fits(self.comp, "%s/test_deconvolve_mmclean_linear-comp.fits" % self.dir)
        if self.persist: export_image_to_fits(self.residual, "%s/test_deconvolve_mmclean_linear-residual.fits" % self.dir)
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist: export_image_to_fits(self.cmodel, "%s/test_deconvolve_mmclean_linear-clean.fits" % self.dir)
        assert numpy.max(self.residual.data) < 3.0
    
    def test_deconvolve_mmclean_linear_noscales(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=self.niter, gain=0.1,
                                                   algorithm='mmclean',
                                                   scales=[0], threshold=0.01, nmoment=2, findpeak='ARL',
                                                   fractional_threshold=0.01, window=self.innerquarter)
        if self.persist: export_image_to_fits(self.comp, "%s/test_deconvolve_mmclean_linear_noscales-comp.fits" % self.dir)
        if self.persist: export_image_to_fits(self.residual, "%s/test_deconvolve_mmclean_linear_noscales-residual.fits" % self.dir)
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist: export_image_to_fits(self.cmodel, "%s/test_deconvolve_mmclean_linear_noscales-clean.fits" % self.dir)
        assert numpy.max(self.residual.data) < 3.0
    
    def test_deconvolve_mmclean_quadratic(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=self.niter, gain=0.1,
                                                   algorithm='mmclean',
                                                   scales=[0, 3, 10], threshold=0.01, nmoment=2, findpeak='ARL',
                                                   fractional_threshold=0.01, window=self.innerquarter)
        if self.persist: export_image_to_fits(self.comp, "%s/test_deconvolve_mmclean_quadratic-comp.fits" % self.dir)
        if self.persist: export_image_to_fits(self.residual, "%s/test_deconvolve_mmclean_quadratic-residual.fits" % self.dir)
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist: export_image_to_fits(self.cmodel, "%s/test_deconvolve_mmclean_quadratic-clean.fits" % self.dir)
        assert numpy.max(self.residual.data) < 3.0
    
    def test_deconvolve_mmclean_quadratic_noscales(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=self.niter, gain=0.1,
                                                   algorithm='mmclean',
                                                   scales=[0], threshold=0.01, nmoment=2, findpeak='ARL',
                                                   fractional_threshold=0.01, window=self.innerquarter)
        if self.persist: export_image_to_fits(self.comp, "%s/test_deconvolve_mmclean_quadratic_noscales-comp.fits" % self.dir)
        if self.persist: export_image_to_fits(self.residual, "%s/test_deconvolve_mmclean_quadratic_noscales-residual.fits" % self.dir)
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist: export_image_to_fits(self.cmodel, "%s/test_deconvolve_mmclean_quadratic_noscales-clean.fits" % self.dir)
        assert numpy.max(self.residual.data) < 3.0

    def test_deconvolve_mmclean_quadratic_psf(self):
        self.comp, self.residual = deconvolve_cube(self.dirty, self.psf, niter=self.niter, gain=0.1,
                                                   algorithm='mmclean',
                                                   scales=[0, 3, 10], threshold=0.01, nmoment=2, findpeak='ARL',
                                                   fractional_threshold=0.01, window=self.innerquarter,
                                                   psf_support=32)
        if self.persist: export_image_to_fits(self.comp, "%s/test_deconvolve_mmclean_quadratic_psf-comp.fits" % self.dir)
        if self.persist: export_image_to_fits(self.residual, "%s/test_deconvolve_mmclean_quadratic_psf-residual.fits" % self.dir)
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist: export_image_to_fits(self.cmodel, "%s/test_deconvolve_mmclean_quadratic_psf-clean.fits" % self.dir)
        assert numpy.max(self.residual.data) < 3.0


if __name__ == '__main__':
    unittest.main()
