""" Unit tests for peeling


"""
import os
import numpy
import unittest

from astropy.coordinates import SkyCoord
import astropy.units as u

from arl.data.data_models import Skycomponent
from arl.data.polarisation import PolarisationFrame

from arl.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility
from arl.calibration.peeling import peel_skycomponent_blockvisibility
from arl.skycomponent.operations import apply_beam_to_skycomponent
from arl.util.testing_support import create_named_configuration, simulate_gaintable, \
    create_low_test_skycomponents_from_gleam, create_low_test_beam
from arl.visibility.base import create_blockvisibility
from arl.imaging import predict_skycomponent_visibility, create_image_from_visibility
from arl.imaging.imaging_context import invert_function
from arl.image.operations import qa_image, export_image_to_fits
from arl.visibility.iterators import vis_timeslices

import logging

log = logging.getLogger(__name__)


class TestCalibrationPeeling(unittest.TestCase):
    
    @unittest.skip("Skip until rewrite")
    def test_peel_skycomponent_blockvisibility(self):
        
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)

        df = 1e6
        frequency = numpy.array([1e8 - df, 1e8, 1e8 + df])
        channel_bandwidth = numpy.array([df, df, df])
    
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        flux = numpy.array([f, 0.8 * f, 0.6 * f])
        phasecentre = SkyCoord(0 * u.deg, -60.0 * u.deg)

        config = create_named_configuration('LOWBD2-CORE')
        peeldirection = SkyCoord(+15 * u.deg, -60.0 * u.deg)
        times = numpy.linspace(-3.0, 3.0, 7) * numpy.pi / 12.0

        # Make the visibility
        vis = create_blockvisibility(config, times, frequency, phasecentre=phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('linear'),
                                     channel_bandwidth=channel_bandwidth)
        vis.data['vis'][...] = 0.0

        # First add in the source to be peeled.
        peel = Skycomponent(direction=peeldirection, frequency=frequency, flux=flux,
                            polarisation_frame=PolarisationFrame("stokesIQUV"))
        vis = predict_skycomponent_visibility(vis, peel)

        # Make a gaintable and apply it to the visibility of the peeling source
        gt = create_gaintable_from_blockvisibility(vis, timeslice='auto')
        vis_slices = vis_timeslices(vis, timeslice='auto')
        gt = simulate_gaintable(gt, phase_error=0.01, amplitude_error=0.01, vis_slices=vis_slices)
        gt.data['gain'] *= 0.3
        vis = apply_gaintable(vis, gt, vis_slices=vis_slices)
        
        # Now create a plausible field using the GLEAM sources
        model = create_image_from_visibility(vis, cellsize=0.001, frequency=frequency,
                                             polarisation_frame=PolarisationFrame('stokesIQUV'))

        bm = create_low_test_beam(model=model)
        sc = create_low_test_skycomponents_from_gleam(flux_limit=1.0,
                                                      polarisation_frame=PolarisationFrame("stokesIQUV"),
                                                      frequency=frequency, kind='cubic',
                                                      phasecentre=phasecentre,
                                                      radius=0.1)
        sc = apply_beam_to_skycomponent(sc, bm)
        # Add in the visibility due to these sources
        vis = predict_skycomponent_visibility(vis, sc)
        assert numpy.max(numpy.abs(vis.vis)) > 0.0

        # Now we can peel
        vis, peel_gts = peel_skycomponent_blockvisibility(vis, peel)

        assert len(peel_gts) == 1
        residual = numpy.max(peel_gts[0].residual)
        assert residual < 0.7, "Peak residual %.6f too large" % (residual)
        
        im, sumwt = invert_function(vis, model, context='timeslice', vis_slices=vis_slices)
        qa = qa_image(im)
        export_image_to_fits(im, '%s/test_peel_skycomponent_residual.fits' % self.dir)

        assert numpy.abs(qa.data['max'] - 14.2) < 1.0, str(qa)


if __name__ == '__main__':
    unittest.main()
