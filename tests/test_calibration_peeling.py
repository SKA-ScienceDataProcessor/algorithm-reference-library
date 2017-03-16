"""Unit tests for peeling

realtimcornwell@gmail.com
"""
import unittest

from arl.calibration.operations import *
from arl.fourier_transforms.ftprocessor import predict_skycomponent_blockvisibility, create_image_from_visibility
from arl.util.testing_support import create_named_configuration, simulate_gaintable, \
    create_low_test_skycomponents_from_gleam, create_low_test_beam
from arl.skycomponent.operations import apply_beam_to_skycomponent
from arl.visibility.operations import create_blockvisibility, copy_visibility
from arl.calibration.peeling import peel_skycomponent_blockvisibility

import logging


log = logging.getLogger(__name__)

class TestCalibrationPeeling(unittest.TestCase):
    
    def test_peel_skycomponent_blockvisibility(self):
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
        vis = create_blockvisibility(config, times, frequency,
                                          channel_bandwidth=channel_bandwidth,
                                          phasecentre=phasecentre, weight=1.0,
                                          polarisation_frame=PolarisationFrame('linear'))
        vis.data['vis'][...]=0.0

        # First add in the source to be peeled.
        peel = Skycomponent(direction=peeldirection, frequency=frequency, flux=flux,
                            polarisation_frame=PolarisationFrame("stokesIQUV"))
        vis = predict_skycomponent_blockvisibility(vis, peel)

        # Make a gaintable and apply it to the visibility of the peeling source
        gt = create_gaintable_from_blockvisibility(vis)
        gt = simulate_gaintable(gt, phase_error=0.01, amplitude_error=0.01)
        gt.data['gain']*=0.3
        vis = apply_gaintable(vis, gt)
        
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
        vis = predict_skycomponent_blockvisibility(vis, sc)

        # Now we can peel
        vis, peel_gts = peel_skycomponent_blockvisibility(vis, peel)

        assert len(peel_gts) == 1
        residual = numpy.max(peel_gts[0].residual)
        assert residual < 0.5, "Peak residual %.6f too large" % (residual)


if __name__ == '__main__':
    unittest.main()
