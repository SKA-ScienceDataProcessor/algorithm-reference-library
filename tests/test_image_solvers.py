"""Unit tests for image solution


"""

import logging
import os
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from arl.data.polarisation import PolarisationFrame
from arl.image.deconvolution import restore_cube
from arl.image.operations import export_image_to_fits
from arl.image.solvers import solve_image
from arl.util.testing_support import create_test_image, create_named_configuration
from arl.visibility.base import create_visibility
from arl.imaging import invert_2d, predict_2d, create_image_from_visibility

log = logging.getLogger(__name__)


class TestImageSolvers(unittest.TestCase):
    def setUp(self):
        self.dir = './test_results'
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        os.makedirs(self.dir, exist_ok=True)
        self.times = (numpy.pi / (12.0)) * numpy.linspace(-3.0, 3.0, 7)
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e6])
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('stokesI'))
        self.vis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_test_image(cellsize=0.0015, phasecentre=self.vis.phasecentre,
                                       frequency=self.frequency)
        self.model.data[self.model.data > 1.0] = 1.0
        self.vis = predict_2d(self.vis, self.model)
        assert numpy.max(numpy.abs(self.vis.vis)) > 0.0
        export_image_to_fits(self.model, '%s/test_solve_skycomponent_model.fits' % (self.dir))
        self.bigmodel = create_image_from_visibility(self.vis, cellsize=0.0015, npixel=512)
        
        
    def test_deconvolve_and_restore_cube_msclean(self):
        self.bigmodel.data *= 0.0
        visres, model, residual = solve_image(self.vis, self.bigmodel, nmajor=3, niter=1000, threshold=0.01,
                                              gain=0.7, psf_support=200, window='quarter', scales=[0, 3, 10, 30],
                                              fractional_threshold=0.1, algorithm = 'msclean')
        export_image_to_fits(model, '%s/test_solve_skycomponent_msclean_solution.fits' % (self.dir))
        export_image_to_fits(residual, '%s/test_solve_skycomponent_msclean_residual.fits' % (self.dir))
        psf, sumwt = invert_2d(self.vis, model, dopsf=True)
        export_image_to_fits(psf, '%s/test_solve_skycomponent_msclean_psf.fits' % (self.dir))
        restored = restore_cube(model=model, psf=psf, residual=residual)
        export_image_to_fits(restored, '%s/test_solve_skycomponent_msclean_restored.fits' % (self.dir))
        assert numpy.max(numpy.abs(residual.data)) < 1.2

    def test_deconvolve_and_restore_cube_hogbom(self):
        self.bigmodel.data *= 0.0
        visres, model, residual = solve_image(self.vis, self.bigmodel, niter=1000, nmajor=5, threshold=0.01,
                                              psf_support=200, window='quarter',
                                              fractional_threshold=0.1, gain=0.1, algorithm = 'hogbom')
        export_image_to_fits(model, '%s/test_solve_skycomponent_hogbom_solution.fits' % (self.dir))
        export_image_to_fits(residual, '%s/test_solve_skycomponent_hogbom_residual.fits' % (self.dir))
        psf, sumwt = invert_2d(self.vis, model, dopsf=True)
        export_image_to_fits(psf, '%s/test_solve_skycomponent_hogbom_psf.fits' % (self.dir))
        restored = restore_cube(model=model, psf=psf, residual=residual)
        export_image_to_fits(restored, '%s/test_solve_skycomponent_hogbom_restored.fits' % (self.dir))
        assert numpy.max(numpy.abs(residual.data)) < 0.5
        
if __name__ == '__main__':
    unittest.main()
