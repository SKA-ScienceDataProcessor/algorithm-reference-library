""" Unit tests for calibration solution


"""
import numpy

import unittest

from astropy.coordinates import SkyCoord
import astropy.units as u

from data_models.memory_data_models import Skycomponent
from data_models.polarisation import PolarisationFrame

from processing_components.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility, gaintable_summary, \
    qa_gaintable
from processing_components.calibration.calibration import solve_gaintable
from processing_components.simulation.testing_support import create_named_configuration, simulate_gaintable
from processing_components.visibility.operations import divide_visibility
from processing_components.visibility.base import copy_visibility, create_blockvisibility
from processing_components.imaging.base import predict_skycomponent_visibility

import logging

log = logging.getLogger(__name__)


class TestCalibrationSolvers(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(180555)
        
    def actualSetup(self, sky_pol_frame='stokesIQUV', data_pol_frame='linear', f=None, vnchan=3):
        self.lowcore = create_named_configuration('LOWBD2', rmax=300.0)
        self.times = (numpy.pi / 43200.0) * numpy.linspace(0.0, 30.0, 3)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, vnchan)
        self.channel_bandwidth = numpy.array(vnchan * [self.frequency[1] - self.frequency[0]])
        
        if f is None:
            f = [100.0, 50.0, -10.0, 40.0]
            
        if sky_pol_frame == 'stokesI':
            f = [100.0]
            
        self.flux = numpy.outer(numpy.array([numpy.power(freq / 1e8, -0.7) for freq in self.frequency]), f)
    
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.comp = Skycomponent(direction=self.compabsdirection, frequency=self.frequency, flux=self.flux,
                                 polarisation_frame=PolarisationFrame(sky_pol_frame))
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                          channel_bandwidth=self.channel_bandwidth, weight=1.0,
                                          polarisation_frame=PolarisationFrame(data_pol_frame))
        self.vis = predict_skycomponent_visibility(self.vis, self.comp)

    def test_solve_gaintable_scalar(self):
        self.actualSetup('stokesI', 'stokesI', f=[100.0])
        gt = create_gaintable_from_blockvisibility(self.vis)
        log.info("Created gain table: %s" % (gaintable_summary(gt)))
        gt = simulate_gaintable(gt, phase_error=10.0, amplitude_error=0.0)
        original = copy_visibility(self.vis)
        self.vis = apply_gaintable(self.vis, gt)
        gtsol = solve_gaintable(self.vis, original, phase_only=True, niter=200)
        residual = numpy.max(gtsol.residual)
        assert residual < 3e-8, "Max residual = %s" % (residual)
        assert numpy.max(numpy.abs(gtsol.gain - 1.0)) > 0.1

    def test_solve_gaintable_scalar_normalise(self):
        self.actualSetup('stokesI', 'stokesI', f=[100.0])
        gt = create_gaintable_from_blockvisibility(self.vis)
        log.info("Created gain table: %s" % (gaintable_summary(gt)))
        gt = simulate_gaintable(gt, phase_error=0.0, amplitude_error=0.1)
        gt.data['gain'] *= 2.0
        original = copy_visibility(self.vis)
        self.vis = apply_gaintable(self.vis, gt)
        gtsol = solve_gaintable(self.vis, original, phase_only=False, niter=200, normalise_gains=True)
        residual = numpy.max(gtsol.residual)
        assert residual < 3e-8, "Max residual = %s" % (residual)
        assert numpy.max(numpy.abs(gtsol.gain - 1.0)) > 0.1

    def test_solve_gaintable_scalar_bandpass(self):
        self.actualSetup('stokesI', 'stokesI', f=[100.0], vnchan=128)
        gt = create_gaintable_from_blockvisibility(self.vis)
        log.info("Created gain table: %s" % (gaintable_summary(gt)))
        gt = simulate_gaintable(gt, phase_error=10.0, amplitude_error=0.01, smooth_channels=8)
        original = copy_visibility(self.vis)
        self.vis = apply_gaintable(self.vis, gt)
        gtsol = solve_gaintable(self.vis, original, phase_only=False, niter=200)
        residual = numpy.max(gtsol.residual)
        assert residual < 3e-8, "Max residual = %s" % (residual)
        assert numpy.max(numpy.abs(gtsol.gain - 1.0)) > 0.1

    def test_solve_gaintable_scalar_pointsource(self):
        self.actualSetup('stokesI', 'stokesI', f=[100.0])
        gt = create_gaintable_from_blockvisibility(self.vis)
        log.info("Created gain table: %s" % (gaintable_summary(gt)))
        gt = simulate_gaintable(gt, phase_error=10.0, amplitude_error=0.0)
        original = copy_visibility(self.vis)
        self.vis = apply_gaintable(self.vis, gt)
        point_vis = divide_visibility(self.vis, original)
        gtsol = solve_gaintable(point_vis, phase_only=True, niter=200)
        residual = numpy.max(gtsol.residual)
        assert residual < 3e-8, "Max residual = %s" % (residual)
        assert numpy.max(numpy.abs(gtsol.gain - 1.0)) > 0.1

    def core_solve(self, spf, dpf, phase_error=0.1, amplitude_error=0.0, leakage=0.0,
                   phase_only=True, niter=200, crosspol=False, residual_tol=1e-6, f=None, vnchan=3):
        if f is None:
            f = [100.0, 50.0, -10.0, 40.0]
        self.actualSetup(spf, dpf, f=f, vnchan=vnchan)
        gt = create_gaintable_from_blockvisibility(self.vis)
        log.info("Created gain table: %s" % (gaintable_summary(gt)))
        gt = simulate_gaintable(gt, phase_error=phase_error, amplitude_error=amplitude_error, leakage=leakage)
        original = copy_visibility(self.vis)
        vis = apply_gaintable(self.vis, gt)
        gtsol = solve_gaintable(self.vis, original, phase_only=phase_only, niter=niter, crosspol=crosspol, tol=1e-6)
        vis = apply_gaintable(vis, gtsol, inverse=True)
        residual = numpy.max(gtsol.residual)
        assert residual < residual_tol, "%s %s Max residual = %s" % (spf, dpf, residual)
        log.debug(qa_gaintable(gt))
        assert numpy.max(numpy.abs(gtsol.gain - 1.0)) > 0.1

    def test_solve_gaintable_vector_phase_only_linear(self):
        self.core_solve('stokesIQUV', 'linear', phase_error=0.1, phase_only=True,
                        f=[100.0, 50.0, 0.0, 0.0])
    
    def test_solve_gaintable_vector_phase_only_circular(self):
        self.core_solve('stokesIQUV', 'circular', phase_error=0.1, phase_only=True,
                        f=[100.0, 0.0, 0.0, 50.0])
    
    def test_solve_gaintable_vector_large_phase_only_linear(self):
        self.core_solve('stokesIQUV', 'linear', phase_error=10.0, phase_only=True,
                        f=[100.0, 50.0, 0.0, 0.0])
    
    def test_solve_gaintable_vector_large_phase_only_circular(self):
        self.core_solve('stokesIQUV', 'circular', phase_error=10.0,
                        phase_only=True, f=[100.0, 0.0, 0.0, 50.0])
    
    def test_solve_gaintable_vector_both_linear(self):
        self.core_solve('stokesIQUV', 'linear', phase_error=0.1, amplitude_error=0.01,
                        phase_only=False, f=[100.0, 50.0, 0.0, 0.0])
    
    def test_solve_gaintable_vector_both_circular(self):
        self.core_solve('stokesIQUV', 'circular', phase_error=0.1, amplitude_error=0.01,
                        phase_only=False, f=[100.0, 0.0, 0.0, 50.0])
    
    def test_solve_gaintable_matrix_both_linear(self):
        self.core_solve('stokesIQUV', 'linear', phase_error=0.1, amplitude_error=0.01,
                        leakage=0.01, residual_tol=1e-3, crosspol=True,
                        phase_only=False, f=[100.0, 50.0, 0.0, 0.0])
    
    def test_solve_gaintable_matrix_both_circular(self):
        self.core_solve('stokesIQUV', 'circular', phase_error=0.1, amplitude_error=0.01,
                        leakage=0.01, residual_tol=1e-3, crosspol=True,
                        phase_only=False, f=[100.0, 0.0, 0.0, 50.0])

    def test_solve_gaintable_matrix_both_circular_channel(self):
        self.core_solve('stokesIQUV', 'circular', phase_error=0.1, amplitude_error=0.01,
                        leakage=0.01, residual_tol=1e-3, crosspol=True, vnchan=4,
                        phase_only=False, f=[100.0, 0.0, 0.0, 50.0])

if __name__ == '__main__':
    unittest.main()
