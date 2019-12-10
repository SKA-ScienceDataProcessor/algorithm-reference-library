"""Functions used to simulate RFI. Developed as part of SP-122/SIM.

The scenario is:
* There is a TV station at a remote location (e.g. Perth), emitting a broadband signal (7MHz) of known power (50kW).
* The emission from the TV station arrives at LOW stations with phase delay and attenuation. Neither of these are
well known but they are probably static.
* The RFI enters LOW stations in a sidelobe of the main beam. Calulations by Fred Dulwich indicate that this
provides attenuation of about 55 - 60dB for a source close to the horizon.
* The RFI enters each LOW station with fixed delay and zero fringe rate (assuming no e.g. ionospheric ducting)
* In tracking a source on the sky, the signal from one station is delayed and fringe-rotated to stop the fringes for one direction on the sky.
* The fringe rotation stops the fringe from a source at the phase tracking centre but phase rotates the RFI, which
now becomes time-variable.
* The correlation data are time- and frequency-averaged over a timescale appropriate for the station field of view.
This averaging decorrelates the RFI signal.
* We want to study the effects of this RFI on statistics of the images: on source and at the pole.
"""

__all__ = ['simulate_DTV', 'create_propagators', 'calculate_averaged_correlation', 'calculate_rfi_at_station',
           'simulate_rfi_block', 'calculate_station_correlation_rfi']


import numpy
from astropy import constants
import astropy.units as u
from astropy.coordinates import SkyCoord

from arl.processing_library.util.array_functions import average_chunks2
from arl.processing_library.util.compass_bearing import calculate_initial_compass_bearing
from arl.processing_library.util.coordinate_support import skycoord_to_lmn, azel_to_hadec, hadec_to_azel
from arl.processing_components.visibility.base import simulate_point

def simulate_DTV(frequency, times, power=50e3, timevariable=False, frequency_variable=False):
    """ Calculate DTV sqrt(power) as a function of time and frequency

    :param frequency: (sample frequencies)
    :param times: sample times (s)
    :param power: DTV emitted power W
    :return: Complex array [ntimes, nchan]
    """
    nchan = len(frequency)
    ntimes = len(times)
    shape = [ntimes, nchan]
    bchan = nchan // 4
    echan = 3 * nchan // 4
    amp = power / (max(frequency) - min(frequency))
    signal = numpy.zeros(shape, dtype='complex')
    if timevariable:
        if frequency_variable:
            sshape = [ntimes, nchan // 2]
            signal[:, bchan:echan] += numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape) \
                                    + 1j * numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape)
        else:
            sshape = [ntimes]
            signal[:, bchan:echan] += numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape) \
                                    + 1j * numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape)
    else:
        if frequency_variable:
            sshape = [nchan // 2]
            signal[:, bchan:echan] += (numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape)
                                   + 1j * numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape))[numpy.newaxis, ...]
        else:
            signal[:, bchan:echan] = amp
    
    return signal


def create_propagators(config, interferer, frequency, attenuation=1e-9):
    """ Create a set of propagators

    :return: Complex array [nants, ntimes]
    """
    nchannels = len(frequency)
    nants = len(config.data['names'])
    interferer_xyz = [interferer.geocentric[0].value, interferer.geocentric[1].value, interferer.geocentric[2].value]
    propagators = numpy.zeros([nants, nchannels], dtype='complex')
    for iant, ant_xyz in enumerate(config.xyz):
        vec = ant_xyz - interferer_xyz
        # This ignores the Earth!
        r = numpy.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        k = 2.0 * numpy.pi * frequency / constants.c.value
        propagators[iant, :] = numpy.exp(- 1.0j * k * r) / r
    return propagators * attenuation


def calculate_rfi_at_station(propagators, emitter):
    """ Calculate the rfi at each station

    :param propagators: [nstations, nchannels]
    :param emitter: [ntimes, nchannels]
    :return: Complex array [nstations, ntimes, nchannels]
    """
    rfi_at_station = emitter[:, numpy.newaxis, ...] * propagators[numpy.newaxis, ...]
    rfi_at_station[numpy.abs(rfi_at_station)<1e-15] = 0.
    return rfi_at_station

def calculate_station_correlation_rfi(rfi_at_station):
    """ Form the correlation from the rfi at the station
    
    :param rfi_at_station:
    :return: Correlation(nant, nants, ntimes, nchan] in Jy
    """
    ntimes, nants, nchan = rfi_at_station.shape
    correlation = numpy.zeros([ntimes, nants, nants, nchan], dtype='complex')
    
    for itime in range(ntimes):
        for chan in range(nchan):
            correlation[itime, ..., chan] = numpy.outer(rfi_at_station[itime, :, chan],
                                                       numpy.conjugate(rfi_at_station[itime, :, chan]))

    return correlation[..., numpy.newaxis] * 1e26


def calculate_averaged_correlation(correlation, time_width, channel_width):
    """ Average the correlation in time and frequency
    
    :param correlation: Correlation(nant, nants, ntimes, nchan]
    :param channel_width: Number of channels to average
    :param time_width: Number of integrations to average
    :return:
    """
    wts = numpy.ones(correlation.shape, dtype='float')
    return average_chunks2(correlation, wts, (time_width, channel_width))[0]


def simulate_rfi_block(bvis, emitter_location, emitter_power=5e4, attenuation=1.0, use_pole=False):
    """ Simulate RFI block

    :param config: ARL telescope Configuration
    :param times: observation times (hour angles)
    :param frequency: frequencies
    :param phasecentre:
    :param emitter_location: EarthLocation of emitter
    :param emitter_power: Power of emitter
    :param attenuation: Attenuation to be applied to signal
    :param use_pole: Set the emitter to nbe at the southern celestial pole
    :return:
    """

    # Calculate the power spectral density of the DTV station: Watts/Hz
    emitter = simulate_DTV(bvis.frequency, bvis.time, power=emitter_power, timevariable=False)
    
    # Calculate the propagators for signals from Perth to the stations in low
    # These are fixed in time but vary with frequency. The ad hoc attenuation
    # is set to produce signal roughly equal to noise at LOW
    propagators = create_propagators(bvis.configuration, emitter_location, frequency=bvis.frequency,
                                     attenuation=attenuation)
    # Now calculate the RFI at the stations, based on the emitter and the propagators
    rfi_at_station = calculate_rfi_at_station(propagators, emitter)

    # Calculate the rfi correlation using the fringe rotation and the rfi at the station
    # [ntimes, nants, nants, nchan, npol]
    bvis.data['vis'][...] = calculate_station_correlation_rfi(rfi_at_station)

    ntimes, nant, _, nchan, npol = bvis.vis.shape

    s2r = numpy.pi / 43200.0
    k = numpy.array(bvis.frequency) / constants.c.to('m s^-1').value
    uvw = bvis.uvw[..., numpy.newaxis] * k

    pole = SkyCoord(ra=+0.0 * u.deg, dec=-90.0 * u.deg, frame='icrs', equinox='J2000')

    if use_pole:
        # Calculate phasor needed to shift from the phasecentre to the pole
        l, m, n = skycoord_to_lmn(pole, bvis.phasecentre)
        phasor = numpy.ones([ntimes, nant, nant, nchan, npol], dtype='complex')
        for chan in range(nchan):
            phasor[:, :, :, chan, :] = simulate_point(uvw[..., chan], l, m)[..., numpy.newaxis]
    
        # Now fill this into the BlockVisibility
        bvis.data['vis'] = bvis.data['vis'] * phasor
    else:
        # We know where the emitter is. Calculate the bearing to the emitter from
        # the site, generate az, el, and convert to ha, dec. ha, dec is static.
        site = bvis.configuration.location
        site_tup = (site.lat.deg, site.lon.deg)
        emitter_tup = (emitter_location.lat.deg, emitter_location.lon.deg)
        az = - calculate_initial_compass_bearing(site_tup, emitter_tup) * numpy.pi / 180.0
        el = 0.0
        hadec = azel_to_hadec(az, el, site.lat.rad)

        # Now step through the time stamps, calculating the effective
        # sky position for the emitter, and performing phase rotation
        # appropriately
        for itime, time in enumerate(bvis.time):
            ra = - hadec[0] + s2r * time
            dec = hadec[1]
            emitter_sky = SkyCoord(ra * u.rad, dec * u.rad)
            l, m, n = skycoord_to_lmn(emitter_sky, bvis.phasecentre)

            phasor = numpy.ones([nant, nant, nchan, npol], dtype='complex')
            for chan in range(nchan):
                phasor[:, :, chan, :] = simulate_point(uvw[itime, ..., chan], l, m)[..., numpy.newaxis]

            # Now fill this into the BlockVisibility
            bvis.data['vis'][itime, ...] = bvis.data['vis'][itime, ...] * phasor

    return bvis
