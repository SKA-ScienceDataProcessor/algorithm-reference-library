"""Unit tests for sky model solution

realtimcornwell@gmail.com
"""

from arl.fourier_transforms.ftprocessor import *
from arl.image.operations import export_image_to_fits
from arl.image.deconvolution import restore_cube
from arl.skymodel.solvers import *
from arl.util.testing_support import *
from arl.visibility.operations import *


class TestSkymodelSolution(unittest.TestCase):
    def setUp(self):
        self.dir = './test_results'
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        os.makedirs(self.dir, exist_ok=True)
        self.times = (numpy.pi / (12.0)) * numpy.linspace(-3.0, 3.0, 7)
        self.frequency = numpy.array([1e8])
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox=2000.0)
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                     weight=1.0, npol=1)
        self.vis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_low_test_image(npixel=1024, cellsize=0.001, phasecentre=self.vis.phasecentre)
        self.beam = create_low_test_beam(model=self.model)
        self.model.data *= self.beam.data
        self.model.data[self.model.data > 1.0] = 1.0
        self.vis = predict_2d(self.vis, self.model)
        assert numpy.max(numpy.abs(self.vis.vis)) > 0.0
    
    def test_deconvolve_and_restore_cube(self):
        self.model.data *= 0.0
        visres, model = solve_skymodel(self.vis, self.model, nmajor=5, threshold=0.01)
        export_image_to_fits(model, '%s/test_solve_skymodel_model.fits' % (self.dir))
        residual = create_empty_image_like(model)
        residual, sumwt = invert_2d(visres, residual)
        residual = normalize_sumwt(residual, sumwt)
        export_image_to_fits(residual, '%s/test_solve_skymodel_residual.fits' % (self.dir))
        psf, _ = invert_2d(self.vis, self.model, dopsf=True)
        restored = restore_cube(model=model, psf=psf, residual=residual)
        export_image_to_fits(restored, '%s/test_solve_skymodel_restored.fits' % (self.dir))
        assert numpy.max(numpy.abs(residual.data)) < 0.5

if __name__ == '__main__':
    unittest.main()
