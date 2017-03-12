"""Unit tests for visibility calibrations

realtimcornwell@gmail.com
"""
import unittest

from arl.calibration.gaintable import *
from arl.fourier_transforms.ftprocessor import *
from arl.util.testing_support import create_named_configuration, simulate_gaintable
from arl.visibility.operations import create_blockvisibility


class TestCalibration(unittest.TestCase):
    
    def setUp(self):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, 3)
        
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])

        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
        
    def actualSetup(self, sky_pol_frame='stokesIQUV', data_pol_frame='linear'):
        self.comp = Skycomponent(direction=self.compabsdirection, frequency=self.frequency, flux=self.flux,
                                 polarisation_frame=Polarisation_Frame(sky_pol_frame))
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                     weight=1.0, polarisation_frame=Polarisation_Frame(data_pol_frame))
        self.vis = predict_skycomponent_blockvisibility(self.vis, self.comp)

    def test_create_gaintable_from_visibility(self):
        for spf, dpf in[('stokesIQUV', 'linear'), ('stokesIQUV', 'circular') ]:
            self.actualSetup(spf, dpf)
            gt = create_gaintable_from_blockvisibility(self.vis)
            log.info("Created gain table: %s" % (gaintable_summary(gt)))
            gt = simulate_gaintable(gt, phase_error=0.1)
            original = copy_visibility(self.vis)
            vis = apply_gaintable(self.vis, gt)
            assert numpy.max(numpy.abs(vis.vis-original.vis)) > 0.0

    def test_apply_gaintable_only(self):
        for spf, dpf in[('stokesIQUV', 'linear'), ('stokesIQUV', 'circular') ]:
            self.actualSetup(spf, dpf)
            gt = create_gaintable_from_blockvisibility(self.vis)
            log.info("Created gain table: %s" % (gaintable_summary(gt)))
            gt = simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.01)
            original = copy_visibility(self.vis)
            vis = apply_gaintable(self.vis, gt)
            error = numpy.max(numpy.abs(vis.vis-original.vis))
            assert error > 10.0, "Error = %f" % (error)

    def test_apply_gaintable_and_inverse_phase_only(self):
        for spf, dpf in[('stokesIQUV', 'linear'), ('stokesIQUV', 'circular') ]:
            self.actualSetup(spf, dpf)
            gt = create_gaintable_from_blockvisibility(self.vis)
            log.info("Created gain table: %s" % (gaintable_summary(gt)))
            gt = simulate_gaintable(gt, phase_error=0.1)
            original = copy_visibility(self.vis)
            vis = apply_gaintable(self.vis, gt)
            vis = apply_gaintable(self.vis, gt, inverse=True)
            error = numpy.max(numpy.abs(vis.vis-original.vis))
            assert error < 1e-12, "Error = %s" % (error)


    def test_apply_gaintable_and_inverse_both(self):
        for spf, dpf in[('stokesIQUV', 'linear'), ('stokesIQUV', 'circular') ]:
            self.actualSetup(spf, dpf)
            gt = create_gaintable_from_blockvisibility(self.vis)
            log.info("Created gain table: %s" % (gaintable_summary(gt)))
            gt = simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.1)
            original = copy_visibility(self.vis)
            vis = apply_gaintable(self.vis, gt)
            vis = apply_gaintable(self.vis, gt, inverse=True)
            error = numpy.max(numpy.abs(vis.vis-original.vis))
            assert error < 1e-12, "Error = %s" % (error)
    
    def test_solve_gaintable_phase_only(self):
        for spf, dpf in[('stokesIQUV', 'linear'), ('stokesIQUV', 'circular') ]:
            self.actualSetup(spf, dpf)
            gt = create_gaintable_from_blockvisibility(self.vis)
            log.info("Created gain table: %s" % (gaintable_summary(gt)))
            gt = simulate_gaintable(gt, phase_error=0.1)
            original = copy_visibility(self.vis)
            vis = apply_gaintable(self.vis, gt)
            gtsol = solve_gaintable(self.vis, original)
            vis = apply_gaintable(vis, gtsol, inverse=True)
            residual = numpy.max(gtsol.residual)
            assert residual < 1e-8, "Max residual = %s" % (residual)


    def test_solve_gaintable_both_big(self):
        for crosspol in [False, True]:
            for spf, dpf in[('stokesIQUV', 'linear'), ('stokesIQUV', 'circular')]:
                self.actualSetup(spf, dpf)
                gt = create_gaintable_from_blockvisibility(self.vis)
                log.info("Created gain table: %s" % (gaintable_summary(gt)))
                gt = simulate_gaintable(gt, phase_error=10.0, amplitude_error=0.1)
                original = copy_visibility(self.vis)
                vis = apply_gaintable(self.vis, gt)
                error = numpy.max(numpy.abs(vis.vis-original.vis))
                assert error > 100, "Error = %s" % (error)
                gtsol = solve_gaintable(self.vis, original, phase_only=False, niter=200, crosspol=crosspol)
                residual = numpy.max(gtsol.residual)
                assert residual < 3e-8, "Crosspol %s %s %s Max residual = %s < 1e-8" % (crosspol, spf, dpf, residual)
    

    def test_solve_gaintable_big_phase(self):
        for spf, dpf in[('stokesIQUV', 'linear'), ('stokesIQUV', 'circular') ]:
            self.actualSetup(spf, dpf)
            gt = create_gaintable_from_blockvisibility(self.vis)
            log.info("Created gain table: %s" % (gaintable_summary(gt)))
            gt = simulate_gaintable(gt, phase_error=10.0, amplitude_error=0.0)
            original = copy_visibility(self.vis)
            vis = apply_gaintable(self.vis, gt)
            gtsol = solve_gaintable(self.vis, original, phase_only=True, niter=200)
            residual = numpy.max(gtsol.residual)
            assert residual < 3e-8, "Max residual = %s" % (residual)

    def test_solve_gaintable_scalar(self):
        self.flux = numpy.array([[self.flux[0,0]]])
        self.frequency = numpy.array([self.frequency[0]])
        self.actualSetup('stokesI', 'stokesI')
        gt = create_gaintable_from_blockvisibility(self.vis)
        log.info("Created gain table: %s" % (gaintable_summary(gt)))
        gt = simulate_gaintable(gt, phase_error=10.0, amplitude_error=0.0)
        original = copy_visibility(self.vis)
        vis = apply_gaintable(self.vis, gt)
        gtsol = solve_gaintable(self.vis, original, phase_only=True, niter=200)
        residual = numpy.max(gtsol.residual)
        assert residual < 3e-8, "Max residual = %s" % (residual)

if __name__ == '__main__':
    unittest.main()
