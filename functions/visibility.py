# Tim Cornwell <realtimcornwell@gmail.com>
#
# Visibility data structure: a Table with columns ['uvw', 'time', 'antenna1', 'antenna2', 'vis', 'weight']
# and an attached attribute which is the frequency of each channel

from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, CartesianRepresentation
from astropy import constants as const

from crocodile.simulate import *
from functions.configuration import Configuration, named_configuration


class Visibility():
    """
    Visibility with uvw, time, a1, a2, vis, weight Columns in
    an astropy Table along with an attribute to hold the frequencies
    and an attribute to hold the direction.

    Visibility is defined to hold an observation with one set of frequencies and one
    direction.

    The data column has vis:[row,nchan,npol], uvw:[row,3]
    """
    def __init__(self):
        self.data = None
        self.frequency = None
        self.phasecentre = None


def visibility_add(fvt1: Visibility, fvt2: Visibility, **kwargs) -> Visibility:
    """

    :param fvt1:
    :param fvt2:
    :param kwargs:
    :return:
    """
    assert len(fvt1.frequency) == len(fvt2.frequency), "Visibility: frequencies should be the same"
    assert numpy.max(numpy.abs(fvt1.frequency - fvt2.frequency)) < 1.0, "Visibility: frequencies should be the same"
    print("visibility.add: adding tables with %d rows and %d rows" % (len(fvt1.data), len(fvt2.data)))
    fvt2rot = phaserotate(fvt2, fvt1.phasecentre)
    fvt = Visibility()
    fvt.data = vstack([fvt1.data, fvt2rot.data], join_type='exact')
    fvt.phasecentre = fvt1.phasecentre
    fvt.frequency = fvt1.frequency
    print(u"visibility.filter: Created table with {0:d} rows".format(len(fvt.data)))
    assert (len(fvt.data) == (len(fvt1.data) + len(fvt2.data))), 'Length of output data table wrong'
    return fvt


def visibility_filter(fvis: Visibility, **kwargs) -> Visibility:
    """

    :param fvis:
    :param kwargs:
    :return:
    """
    print("Visibility.filter: not implemented yet")
    return fvis


def visibility_from_array(uvw: numpy.array, time: numpy.array, freq: numpy.array, antenna1: numpy.array,
                          antenna2: numpy.array, vis: numpy.array, weight: numpy.array,
                          phasecentre: SkyCoord, meta: dict, **kwargs) -> Visibility:
    """

    :param uvw:
    :param time:
    :param freq:
    :param antenna1:
    :param antenna2:
    :param vis:
    :param weight:
    :type phasecentre: SkyCoord
    :type meta: object
    """
    nrows = time.shape[0]
    assert uvw.shape[0] == nrows, "Discrepancy in number of rows in uvw"
    assert len(antenna1) == nrows, "Discrepancy in number of rows in antenna1"
    assert len(antenna2) == nrows, "Discrepancy in number of rows in antenna2"
    assert vis.shape[0] == nrows, "Discrepancy in number of rows for vis"
    assert len(freq) == vis.shape[1], "Discrepancy between frequencies and number of channels"
    assert weight.shape[0] == nrows, "Discrepancy in number of rows"
    vt = Visibility()
    vt.data = Table(data=[uvw, time, antenna1, antenna2, vis, weight],
                    names=['uvw', 'time', 'antenna1', 'antenna2', 'vis', 'weight'], meta=meta)
    vt.frequency = freq
    vt.phasecentre = phasecentre
    return vt


def simulate(config: Configuration, times: numpy.array, freq: numpy.array, weight: float = 1.0,
             phasecentre: SkyCoord = None, meta: dict = None, **kwargs) -> Visibility:
    """
    Creat a vistable from Configuration, hour angles, and direction of source
    :param meta:
    :param config: Configuration of antennas
    :param times: hour angles
    :param freq: frequencies (Hz]
    :param weight: weight of a single sample
    :param phasecentre: phasecentre of observation
    :rtype: object
    """
    assert phasecentre is not None, "Must specify phase centre"
    nch = len(freq)
    npol = kwargs.get("npol", 4)
    ants_xyz = config.data['xyz']
    nants = len(config.data['names'])
    nbaselines = int(nants * (nants - 1) / 2)
    ntimes = len(times)
    nrows = nbaselines * ntimes
    row = 0
    rvis = numpy.zeros([nrows, nch, npol], dtype='complex')
    rweight = weight * numpy.ones([nrows, nch, npol])
    rtimes = numpy.zeros([nrows])
    rantenna1 = numpy.zeros([nrows], dtype='int')
    rantenna2 = numpy.zeros([nrows], dtype='int')
    for ha in times:
        rtimes[row:row + nbaselines] = ha * 43200.0 / numpy.pi
        for a1 in range(nants):
            for a2 in range(a1 + 1, nants):
                rantenna1[row] = a1
                rantenna2[row] = a2
                row += 1
    ruvw = xyz_to_baselines(ants_xyz, times, phasecentre.dec)
    print(u"visibility.simulate: Created {0:d} rows".format(nrows))
    vt = visibility_from_array(ruvw, rtimes, freq, rantenna1, rantenna2, rvis, rweight, phasecentre, meta)
    return vt

def phaserotate(vt: Visibility, newphasecentre: SkyCoord, **kwargs) -> numpy.array:
    """
    Phase rotate from the current phase centre to a new phase centre
    """
    fromdc = vt.phasecentre.represent_as(CartesianRepresentation)
    todc = newphasecentre.represent_as(CartesianRepresentation)
    if numpy.abs(fromdc.x - todc.x) > 1e-15 or numpy.abs(fromdc.y - todc.y) > 1e-15:
        print('visibility.phaserotate: Phase rotation from %s to %s' % (vt.phasecentre, newphasecentre))
        nchan = vt.data['vis'].shape[1]
        npol = vt.data['vis'].shape[2]
        for channel in range(nchan):
            uvw = vt.data['uvw'] * (vt.frequency[channel] / const.c).value
            phasor = simulate_point(uvw, todc.y, todc.z)*numpy.conj(simulate_point(uvw, fromdc.y, fromdc.z))
            for pol in range(npol):
                print('visibility.phaserotate: Phaserotating visibility for channel %d, polarisation %d' % (channel, pol))
                vt.data['vis'][:, channel, pol] = vt.data['vis'][:, channel, pol] * phasor
    # TODO: rotate uvw as well!!!

    vt.phasecentre = newphasecentre
    return vt


def visibilitysum(vt: Visibility, direction: SkyCoord, **kwargs) -> numpy.array:
    """
    Direct Fourier summation in a given direction
    """
    dc = direction.represent_as(CartesianRepresentation)
    nchan = vt.data['vis'].shape[1]
    npol = vt.data['vis'].shape[2]
    flux = numpy.zeros([nchan, npol])
    weight = numpy.zeros([nchan, npol])
    for channel in range(nchan):
        uvw = vt.data['uvw'] * (vt.frequency[channel] / const.c).value
        phasor = simulate_point(uvw, dc.y, dc.z)
        for pol in range(npol):
            print('imaging.visibilitysum: Summing visibility for channel %d, polarisation %d' % (channel, pol))
            flux[channel, pol] = flux[channel, pol] + \
                                 numpy.sum(numpy.real(vt.data['vis'][:, channel, pol] *
                                                      vt.data['weight'][:, channel, pol] * phasor))
            weight[channel, pol] = weight[channel, pol] + numpy.sum(vt.data['weight'][:, channel, pol])
    flux[weight > 0.0] = flux[weight > 0.0] / weight[weight > 0.0]
    flux[weight <= 0.0] = 0.0
    return flux, weight

if __name__ == '__main__':
    config = named_configuration('VLAA')
    times1 = numpy.arange(-3.0, 0.0, 3.0 / 60.0) * numpy.pi / 12.0
    times2 = numpy.arange(0.0, +3.0, 3.0 / 60.0) * numpy.pi / 12.0
    freq1 = numpy.arange(5e6, 150.0e6, 1e7)
    freq2 = numpy.arange(6e6, 150.0e6, 1e7)
    phasecentre1 = SkyCoord(ra='00h42m30s', dec='+41d12m00s', frame='icrs')
    phasecentre2 = SkyCoord(ra='04h56m10s', dec='+63d00m00s', frame='icrs')
    vt1 = simulate(config, times1, freq1, weight=1.0, phasecentre=phasecentre1)
    vt2 = simulate(config, times2, freq1, weight=1.0, phasecentre=phasecentre2)
    vtsum = visibility_add(vt1, vt2)
    print(vtsum.data)
    print(vtsum.frequency)
    print(numpy.unique(vtsum.data['time']))
