"""Unit tests for visibility operations


"""

import unittest

import numpy

from numpy.testing import assert_allclose

from astropy.coordinates import SkyCoord
import astropy.units as u

from arl.data.data_models import Skycomponent
from arl.data.polarisation import PolarisationFrame
from arl.util.testing_support import create_named_configuration
from arl.imaging import predict_skycomponent_visibility
from arl.visibility.coalesce import convert_blockvisibility_to_visibility
from arl.visibility.operations import append_visibility, qa_visibility, \
    sum_visibility
from arl.visibility.base import copy_visibility, create_visibility, create_blockvisibility, create_visibility_from_rows,\
    phaserotate_visibility


class TestVisibilityOperations(unittest.TestCase):
    def setUp(self):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, 3)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        #        self.compabsdirection = SkyCoord(ra=+182 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox='J2000')
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)
        self.comp = Skycomponent(direction=self.compreldirection, frequency=self.frequency, flux=self.flux)
    
    def test_create_visibility(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,  phasecentre=self.phasecentre,
                                     weight=1.0)
        assert self.vis.nvis == len(self.vis.time)
        assert self.vis.nvis == len(self.vis.frequency)

    def test_create_visibility_polarisation(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("linear"))
        assert self.vis.nvis == len(self.vis.time)
        assert self.vis.nvis == len(self.vis.frequency)

    def test_create_visibility_from_rows(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0)
        rows = self.vis.time > 150.0
        for makecopy in [True, False]:
            selected_vis = create_visibility_from_rows(self.vis, rows, makecopy=makecopy)
            assert selected_vis.nvis == numpy.sum(numpy.array(rows))
            
    def test_create_visibility_time(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                          weight=1.0, channel_bandwidth=self.channel_bandwidth)
        assert self.vis.nvis == len(self.vis.time)

    def test_convert_blockvisibility(self):
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                          weight=1.0, channel_bandwidth=self.channel_bandwidth)
        vis = convert_blockvisibility_to_visibility(self.vis)
        assert vis.nvis == len(vis.time)
        assert numpy.unique(vis.time).size == self.vis.time.size

    def test_create_visibility_from_rows_makecopy(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                          weight=1.0, channel_bandwidth=self.channel_bandwidth)
        rows = self.vis.time > 150.0
        for makecopy in [True, False]:
            selected_vis = create_visibility_from_rows(self.vis, rows, makecopy=makecopy)
            assert selected_vis.nvis == numpy.sum(numpy.array(rows))



    def test_append_visibility(self):
            self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                         channel_bandwidth=self.channel_bandwidth,  phasecentre=self.phasecentre,
                                         weight=1.0)
            othertimes = (numpy.pi / 43200.0) * numpy.arange(300.0, 600.0, 30.0)
            self.othervis = create_visibility(self.lowcore, othertimes, self.frequency,
                                              channel_bandwidth=self.channel_bandwidth,  phasecentre=self.phasecentre,
                                              weight=1.0)
            self.vis = append_visibility(self.vis, self.othervis)
            assert self.vis.nvis == len(self.vis.time)
            assert self.vis.nvis == len(self.vis.frequency)

    def test_copy_visibility(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("stokesIQUV"))
        vis = copy_visibility(self.vis)
        self.vis.data['vis'] = 0.0
        vis.data['vis'] = 1.0
        assert (vis.data['vis'][0,0].real == 1.0)
        assert (self.vis.data['vis'][0,0].real == 0.0)
    
    def test_visibilitysum(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("stokesIQUV"))
        self.vismodel = predict_skycomponent_visibility(self.vis, self.comp)
        # Sum the visibilities in the correct_visibility direction. This is limited by numerical precision
        summedflux, weight = sum_visibility(self.vismodel, self.compreldirection)
        assert_allclose(self.flux, summedflux, rtol=1e-7)
    
    def test_phase_rotation_identity(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("stokesIQUV"))
        self.vismodel = predict_skycomponent_visibility(self.vis, self.comp)
        newphasecenters = [SkyCoord(182, -35, unit=u.deg), SkyCoord(182, -30, unit=u.deg),
                           SkyCoord(177, -30, unit=u.deg), SkyCoord(176, -35, unit=u.deg),
                           SkyCoord(216, -35, unit=u.deg), SkyCoord(180, -70, unit=u.deg)]
        for newphasecentre in newphasecenters:
            # Phase rotating back should not make a difference
            original_vis = self.vismodel.vis
            original_uvw = self.vismodel.uvw
            rotatedvis = phaserotate_visibility(phaserotate_visibility(self.vismodel, newphasecentre, tangent=False),
                                                self.phasecentre, tangent=False)
            assert_allclose(rotatedvis.uvw, original_uvw, rtol=1e-7)
            assert_allclose(rotatedvis.vis, original_vis, rtol=1e-7)
    
    def test_phase_rotation(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("stokesIQUV"))
        self.vismodel = predict_skycomponent_visibility(self.vis, self.comp)
        # Predict visibilities with new phase centre independently
        ha_diff = -(self.compabsdirection.ra - self.phasecentre.ra).to(u.rad).value
        vispred = create_visibility(self.lowcore, self.times + ha_diff, self.frequency,
                                    channel_bandwidth=self.channel_bandwidth,
                                    phasecentre=self.compabsdirection, weight=1.0,
                                    polarisation_frame=PolarisationFrame("stokesIQUV"))
        vismodel2 = predict_skycomponent_visibility(vispred, self.comp)
        
        # Should yield the same results as rotation
        rotatedvis = phaserotate_visibility(self.vismodel, newphasecentre=self.compabsdirection, tangent=False)
        assert_allclose(rotatedvis.vis, vismodel2.vis, rtol=1e-7)
        assert_allclose(rotatedvis.uvw, vismodel2.uvw, rtol=1e-7)

    def test_phase_rotation_inverse(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("stokesIQUV"))
        self.vismodel = predict_skycomponent_visibility(self.vis, self.comp)
        there = SkyCoord(ra=+250.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        # Phase rotating back should not make a difference
        original_vis = self.vismodel.vis
        original_uvw = self.vismodel.uvw
        rotatedvis = phaserotate_visibility(phaserotate_visibility(self.vismodel, there, tangent=False,
                                                                   inverse=True),
                                            self.phasecentre, tangent=False, inverse=True)
        assert_allclose(rotatedvis.uvw, original_uvw, rtol=1e-7)
        assert_allclose(rotatedvis.vis, original_vis, rtol=1e-7)

    def test_qa(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("stokesIQUV"))
        self.vismodel = predict_skycomponent_visibility(self.vis, self.comp)
        qa = qa_visibility(self.vis, context='test_qa')
        self.assertAlmostEqual(qa.data['maxabs'], 100.0, 7)
        self.assertAlmostEqual(qa.data['medianabs'], 11.0, 7)
        assert qa.context == 'test_qa'


if __name__ == '__main__':
    unittest.main()
