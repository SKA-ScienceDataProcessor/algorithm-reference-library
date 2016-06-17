# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
import numpy as numpy

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.table import Table, Row, Column, MaskedColumn, TableColumns, TableFormatter
from astropy.nddata import NDData

from functions.fconfiguration import fconfiguration
from functions.fcontext import fcontext
from crocodile.simulate import *


class fvistable(Table):
    """Visibility class with uvw, time, a1, a2, vis, weight columns
    """

    def __init__(self):
        pass

    def __construct(self, uvw: NDData, time: NDData, freq: NDData, antenna1: NDData, antenna2: NDData,
                    vis: NDData, weight: NDData, meta: dict, context: fcontext, **kwargs):
        nrows = time.shape[0]
        assert uvw.shape[0] == nrows, "Discrepancy in number of rows"
        assert len(antenna1) == nrows, "Discrepancy in number of rows"
        assert len(antenna2) == nrows, "Discrepancy in number of rows"
        assert vis.shape[0] == nrows, "Discrepancy in number of rows"
        assert len(freq) == vis.shape[1], "Discrepancy in frequencies"
        assert weight.shape[0] == nrows, "Discrepancy in number of rows"
        super(fvistable, self).__init__(data=[uvw, time, antenna1, antenna2, vis, weight],
                                        names=['uvw', 'time', 'antenna1', 'antenna2', 'vis', 'weight'],
                                        meta=meta)
        self.freq = freq
        return self

    def observe(self, config: fconfiguration, times: numpy.array, freq: numpy.array, weight: float = 1,
                direction: SkyCoord = None, meta: dict = None, context: fcontext = None, **kwargs):
        """
        Creat a vistable from configuration, hour angles, and direction of source
        :param config: Configuration of antennas
        :param times: hour angles
        :param freq: frequencies (Hz]
        :param weight: weight of a single sample
        :param direction: direction of source
        :rtype: object
        """
        assert direction != None, "Must specify direction"
        nch = len(freq)
        ants_xyz = config['xyz']
        nants = len(config['names'])
        nbaselines = int(nants * (nants - 1) / 2)
        ntimes = len(times)
        nrows = nbaselines * ntimes
        row = 0
        rvis = numpy.zeros([nrows, nch, 4], dtype='complex')
        rweight = weight * numpy.zeros([nrows, nch, 4], dtype='float')
        rtimes = numpy.zeros([nrows])
        rantenna1 = numpy.zeros([nrows], dtype='int')
        rantenna2 = numpy.zeros([nrows], dtype='int')
        for ha in times:
            for a1 in range(nants):
                for a2 in range(a1 + 1, nants):
                    rtimes[row] = (ha - times[0]) * 43200.0 / numpy.pi
                    rantenna1[row] = a1
                    rantenna2[row] = a2
                    row += 1
        ruvw = xyz_to_baselines(ants_xyz, times, direction.dec)
        print(u"Created {0:d} rows".format(nrows))
        self.meta = meta
        return self.__construct(ruvw, rtimes, freq, rantenna1, rantenna2, rvis, rweight, context, kwargs)


if __name__ == '__main__':
    import os

    config = fconfiguration()
    config = config.fromname('VLAA')
    times = numpy.arange(-3.0, +3.0, 3.0 / 60.0) * numpy.pi / 12.0
    freq = numpy.arange(5e6, 150.0e6, 1e7)
    direction = SkyCoord('00h42m30s', '-41d12m00s', frame='icrs')
    vt = fvistable()
    vt.observe(config, times, freq, weight=1.0, direction=direction)
    print(vt)
    print(vt.freq)
