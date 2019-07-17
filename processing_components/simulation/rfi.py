import numpy
from astropy import constants
import astropy.units as u
from astropy.coordinates import SkyCoord

from processing_library.util.array_functions import average_chunks2
from processing_library.util.coordinate_support import xyz_to_uvw, skycoord_to_lmn

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


def add_noise(visibility, bandwidth, int_time):
    """Determine noise rms per visibility

    :returns: visibility with noise added
    """
    # The specified sensitivity (effective area / T_sys) is roughly 610 m ^ 2 / K in the range 160 - 200MHz
    # sigma_vis = 2 k T_sys / (area * sqrt(tb)) = 2 k 512 / (610 * sqrt(tb)
    sens = 610
    k_b = 1.38064852e-23
    bt = bandwidth * int_time
    sigma = 2 * 1e26 * k_b / ((sens / 512) * (numpy.sqrt(bt)))
    print("RMS noise per sample = %g Jy" % sigma)
    sshape = visibility.shape
    visibility += numpy.random.normal(0.0, sigma, sshape) + 1j * numpy.random.normal(0.0, sigma, sshape)
    return visibility


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

    :param propagators:
    :param emitter:
    :return: Complex array [nants, ntimes, nchannels]
    """
    rfi_at_station = emitter[numpy.newaxis, :, :] * propagators[:, numpy.newaxis, :]
    rfi_at_station[numpy.abs(rfi_at_station)<1e-15] = 0.
    return rfi_at_station


def calculate_station_fringe_rotation(ants_xyz, times, frequency, phasecentre, pole):
    """Calculate fringe rotation vectors
    
    :param ants_xyz:
    :param times:
    :param frequency:
    :param phasecentre:
    :param pole:
    :return: phasor, uvw
    """
    
    uvw = xyz_to_uvw(ants_xyz, times, phasecentre.dec.rad)
    nants, _ = uvw.shape
    ntimes = len(times)
    uvw = uvw.reshape([nants, 3, ntimes])
    uvw = numpy.transpose(uvw, [0, 2, 1])
    lmn = skycoord_to_lmn(phasecentre, pole)
    delay = numpy.dot(uvw, lmn)
    nchan = len(frequency)
    phase = numpy.zeros([nants, ntimes, nchan])
    for ant in range(nants):
        for chan in range(nchan):
            phase[ant, :, chan] = delay[ant] * frequency[chan] / constants.c.value
    phase[...] -= phase[0, :, :][numpy.newaxis,...]
    return numpy.exp(2.0 * numpy.pi * 1j * phase), uvw


def calculate_station_correlation_rfi(fringe_rotation, rfi_at_station):
    """ Form the correlation from the rfi at the station and the fringe rotation
    
    :param fringe_rotation:
    :param rfi_at_station:
    :return: Correlation(nant, nants, ntimes, nchan] in Jy
    """
    phased_rotated_rfi_at_station = fringe_rotation * rfi_at_station
    nants, ntimes, nchan = fringe_rotation.shape
    correlation = numpy.zeros([nants, nants, ntimes, nchan], dtype='complex')
    
    for time in range(ntimes):
        for chan in range(nchan):
            correlation[..., time, chan] = numpy.outer(phased_rotated_rfi_at_station[..., time, chan],
                                                       numpy.conjugate(phased_rotated_rfi_at_station[..., time, chan]))
    return correlation * 1e26


def calculate_averaged_correlation(correlation, channel_width, time_width):
    """ Average the correlation in time and frequency
    
    :param correlation: Correlation(nant, nants, ntimes, nchan]
    :param channel_width: Number of channels to average
    :param time_width: Number of integrations to average
    :return:
    """
    wts = numpy.ones(correlation.shape, dtype='float')
    return average_chunks2(correlation, wts, (channel_width, time_width))[0]


def simulate_rfi_block(config, times, frequency, phasecentre, emitter_location, emitter_power=5e4,
                       attenuation=1.0):
    """ Simulate RFI block

    :param config: ARL telescope Configuration
    :param times: observation times (hour angles)
    :param frequency: frequencies
    :param phasecentre:
    :param emitter_location: EarthLocation of emitter
    :param emitter_power: Power of emitter
    :param attenuation: Attenuation to be applied to signal
    :return:
    """
    # Calculate the power spectral density of the DTV station: Watts/Hz
    emitter = simulate_DTV(frequency, times, power=emitter_power, timevariable=False)
    
    # Calculate the propagators for signals from Perth to the stations in low
    # These are fixed in time but vary with frequency. The ad hoc attenuation
    # is set to produce signal roughly equal to noise at LOW
    propagators = create_propagators(config, emitter_location, frequency=frequency,
                                     attenuation=attenuation)
    # Now calculate the RFI at the stations, based on the emitter and the propagators
    rfi_at_station = calculate_rfi_at_station(propagators, emitter)
    
    # Station fringe rotation: shape [nants, ntimes, nchan] complex phasor to be applied to
    # reference to the pole.
    pole = SkyCoord(ra=+0.0 * u.deg, dec=-90.0 * u.deg, frame='icrs', equinox='J2000')
    ha = times * numpy.pi / 43200.0

    fringe_rotation, uvw = calculate_station_fringe_rotation(config.xyz, ha, frequency, phasecentre, pole)
    
    # Calculate the rfi correlationm using the fringe rotation and the rfi at the station
    # [nants, nants, ntimes, nchan]
    correlation = calculate_station_correlation_rfi(fringe_rotation, rfi_at_station)
    
    return correlation, uvw
