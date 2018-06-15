""" Unit tests for visibility calibration


"""

import numpy
import logging
import unittest

from astropy.coordinates import SkyCoord
import astropy.units as u

from data_models.memory_data_models import Skycomponent, GainTable
from data_models.polarisation import PolarisationFrame

from processing_components.calibration.operations import gaintable_summary, apply_gaintable, create_gaintable_from_blockvisibility, \
    create_gaintable_from_rows
from processing_components.simulation.testing_support import create_named_configuration, simulate_gaintable
from processing_components.visibility.base import copy_visibility, create_blockvisibility
from processing_components.imaging.base import predict_skycomponent_visibility

log = logging.getLogger(__name__)


class TestCalibrationOperations(unittest.TestCase):
    
    def setUp(self):
        pass
        
    def actualSetup(self, sky_pol_frame='stokesIQUV', data_pol_frame='linear'):
        self.lowcore = create_named_configuration('LOWBD2', rmax=300.0)
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        vnchan = 3
        self.frequency = numpy.linspace(1.0e8, 1.1e8, vnchan)
        self.channel_bandwidth = numpy.array(vnchan * [self.frequency[1] - self.frequency[0]])
    
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
    
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        if sky_pol_frame == 'stokesI':
            self.flux = self.flux[:,0][:, numpy.newaxis]
            
        self.comp = Skycomponent(direction=self.compabsdirection, frequency=self.frequency, flux=self.flux,
                                 polarisation_frame=PolarisationFrame(sky_pol_frame))
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                          phasecentre=self.phasecentre,
                                          channel_bandwidth=self.channel_bandwidth,
                                          weight=1.0,
                                          polarisation_frame=PolarisationFrame(data_pol_frame))
        self.vis = predict_skycomponent_visibility(self.vis, self.comp)

    def test_create_gaintable_from_visibility(self):
        for spf, dpf in[('stokesI', 'stokesI'), ('stokesIQUV', 'linear'), ('stokesIQUV', 'circular')]:
            self.actualSetup(spf, dpf)
            gt = create_gaintable_from_blockvisibility(self.vis, timeslice='auto')
            log.info("Created gain table: %s" % (gaintable_summary(gt)))
            gt = simulate_gaintable(gt, phase_error=1.0)
            original = copy_visibility(self.vis)
            vis = apply_gaintable(self.vis, gt)
            assert numpy.max(numpy.abs(original.vis)) > 0.0
            assert numpy.max(numpy.abs(vis.vis)) > 0.0
            assert numpy.max(numpy.abs(vis.vis - original.vis)) > 0.0


    def test_create_gaintable_from_other(self):
        for timeslice in [10.0, 'auto', 1e5]:
            for spf, dpf in[('stokesIQUV', 'linear')]:
                self.actualSetup(spf, dpf)
                gt = create_gaintable_from_blockvisibility(self.vis, timeslice=timeslice)
                log.info("Created gain table: %s" % (gaintable_summary(gt)))
                new_gt = GainTable(data=gt.data)
                assert new_gt.data.shape == gt.data.shape


    def test_create_gaintable_from_visibility_interval(self):
        for timeslice in [10.0, 'auto', 1e5]:
            for spf, dpf in[('stokesIQUV', 'linear')]:
                self.actualSetup(spf, dpf)
                gt = create_gaintable_from_blockvisibility(self.vis, timeslice=timeslice)
                log.info("Created gain table: %s" % (gaintable_summary(gt)))
                gt = simulate_gaintable(gt, phase_error=1.0)
                original = copy_visibility(self.vis)
                vis = apply_gaintable(self.vis, gt)
                assert numpy.max(numpy.abs(original.vis)) > 0.0
                assert numpy.max(numpy.abs(vis.vis)) > 0.0
                assert numpy.max(numpy.abs(vis.vis - original.vis)) > 0.0

    def test_apply_gaintable_only(self):
        for spf, dpf in[('stokesI', 'stokesI'), ('stokesIQUV', 'linear'), ('stokesIQUV', 'circular')]:
            self.actualSetup(spf, dpf)
            gt = create_gaintable_from_blockvisibility(self.vis, timeslice='auto')
            log.info("Created gain table: %s" % (gaintable_summary(gt)))
            gt = simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.01)
            original = copy_visibility(self.vis)
            vis = apply_gaintable(self.vis, gt)
            error = numpy.max(numpy.abs(vis.vis - original.vis))
            assert error > 10.0, "Error = %f" % (error)

    def test_apply_gaintable_and_inverse_phase_only(self):
        for spf, dpf in[('stokesI', 'stokesI'), ('stokesIQUV', 'linear'), ('stokesIQUV', 'circular')]:
            self.actualSetup(spf, dpf)
            gt = create_gaintable_from_blockvisibility(self.vis, timeslice='auto')
            log.info("Created gain table: %s" % (gaintable_summary(gt)))
            gt = simulate_gaintable(gt, phase_error=0.1)
            original = copy_visibility(self.vis)
            vis = apply_gaintable(self.vis, gt)
            vis = apply_gaintable(self.vis, gt, inverse=True)
            error = numpy.max(numpy.abs(vis.vis - original.vis))
            assert error < 1e-12, "Error = %s" % (error)

    def test_apply_gaintable_and_inverse_both(self):
        for spf, dpf in[('stokesI', 'stokesI'), ('stokesIQUV', 'linear'), ('stokesIQUV', 'circular')]:
            self.actualSetup(spf, dpf)
            gt = create_gaintable_from_blockvisibility(self.vis, timeslice='auto')
            log.info("Created gain table: %s" % (gaintable_summary(gt)))
            gt = simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.1)
            original = copy_visibility(self.vis)
            vis = apply_gaintable(self.vis, gt)
            vis = apply_gaintable(self.vis, gt, inverse=True)
            error = numpy.max(numpy.abs(vis.vis - original.vis))
            assert error < 1e-12, "Error = %s" % (error)

    def test_apply_gaintable_null(self):
        for spf, dpf in[('stokesI', 'stokesI'), ('stokesIQUV', 'linear'), ('stokesIQUV', 'circular')]:
            self.actualSetup(spf, dpf)
            gt = create_gaintable_from_blockvisibility(self.vis, timeslice='auto')
            gt.data['gain']*=0.0
            original = copy_visibility(self.vis)
            vis = apply_gaintable(self.vis, gt, inverse=True)
            error = numpy.max(numpy.abs(vis.vis[:,0,1,...] - original.vis[:,0,1,...]))
            assert error < 1e-12, "Error = %s" % (error)

    def test_create_gaintable_from_rows_makecopy(self):
        self.actualSetup('stokesIQUV', 'linear')
        gt = create_gaintable_from_blockvisibility(self.vis, timeslice='auto')
        rows = gt.time > 150.0
        for makecopy in [True, False]:
            selected_gt = create_gaintable_from_rows(gt, rows, makecopy=makecopy)
            assert selected_gt.ntimes == numpy.sum(numpy.array(rows))



if __name__ == '__main__':
    unittest.main()
