""" Unit tests for RFI simulation

"""

import logging
import unittest

import astropy.units as u
import numpy
import numpy.testing

from astropy.coordinates import SkyCoord, EarthLocation

from data_models.polarisation import PolarisationFrame
from processing_components.simulation.configurations import create_named_configuration
from processing_components.simulation.rfi import create_propagators, calculate_averaged_correlation, \
    calculate_rfi_at_station, calculate_station_correlation_rfi, calculate_station_fringe_rotation, \
    simulate_DTV, add_noise
from processing_components.simulation.testing_support import create_test_image
from processing_components.skycomponent.operations import create_skycomponent
from processing_components.visibility.base import create_blockvisibility
from processing_library.image.operations import create_image

log = logging.getLogger(__name__)


class TestRFISim(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_rfi(self):
        
        sample_freq = 3e4
        nchannels = 1000
        frequency = 170.5e6 + numpy.arange(nchannels) * sample_freq
    
        ntimes = 100
        integration_time = 0.5
        times = numpy.arange(ntimes) * integration_time
    
        phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
        pole = SkyCoord(ra=+0.0 * u.deg, dec=-90.0 * u.deg, frame='icrs', equinox='J2000')
    
        # Perth from Google for the moment
        perth = EarthLocation(lon="115.8605", lat="-31.9505", height=0.0)
    
        rmax = 1000.0
        low = create_named_configuration('LOWR3', rmax=rmax)
        antskip = 33
        low.data = low.data[::antskip]
        nants = len(low.names)
    
        # Calculate the power spectral density of the DTV station: Watts/Hz
        emitter = simulate_DTV(frequency, times, power=50e3, timevariable=False)
        numpy.testing.assert_almost_equal(numpy.max(numpy.abs(emitter)), 0.00166834)
        assert emitter.shape == (ntimes, nchannels)
    
        # Calculate the propagators for signals from Perth to the stations in low
        # These are fixed in time but vary with frequency. The ad hoc attenuation
        # is set to produce signal roughly equal to noise at LOW
        attenuation = 1.0
        propagators = create_propagators(low, perth, frequency=frequency, attenuation=attenuation)
        assert propagators.shape == (nants, nchannels), propagators.shape

        # Now calculate the RFI at the stations, based on the emitter and the propagators
        rfi_at_station = calculate_rfi_at_station(propagators, emitter)
        assert rfi_at_station.shape == (nants, ntimes, nchannels), rfi_at_station.shape
    
        # Station fringe rotation: shape [nants, ntimes, nchan] complex phasor to be applied to
        # reference to the pole.
        fringe_rotation, uvw = calculate_station_fringe_rotation(low.xyz, times, frequency, phasecentre, pole)
        numpy.testing.assert_allclose(numpy.abs(fringe_rotation), 1.0)

        # Calculate the rfi correlationm using the fringe rotation and the rfi at the station
        # [nants, nants, ntimes, nchan]
        correlation = calculate_station_correlation_rfi(fringe_rotation, rfi_at_station)
        assert correlation.shape == (nants, nants, ntimes, nchannels), correlation.shape

if __name__ == '__main__':
    unittest.main()
