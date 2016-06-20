# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
from collections import namedtuple
import numpy as numpy

from astropy.table import Table, vstack, Row, Column, MaskedColumn, TableColumns, TableFormatter


def fgaintab():
    """
    Gain table with time, antenna, frequency columns.
    """
    return namedtuple('fgaintab', ['data'])


def fgaintab_filter(fg: fgaintab, **kwargs):
    print("fgaintab: No filter implemented yet")
    return fg



def fgaintab_add(fgt1: fgaintab, fgt2: fgaintab, **kwargs):
    assert len(fgt1.frequency) == len(fgt2.frequency), "fgaintab: frequencies should be the same"
    assert numpy.max(numpy.abs(fgt1.frequency - fgt2.frequency)) < 1.0, "fgaintab: frequencies should be the same"
    print("fgaintab: adding tables with %d rows and %d rows" % (len(fgt1.data), len(fgt2.data)))
    fgt = fgaintab()
    fgt.data = vstack([fgt1.data, fgt2.data], join_type='exact')
    print(u"Created table with {0:d} rows".format(len(fgt.data)))
    assert(len(fgt.data)==(len(fgt1.data)+len(fgt2.data))), 'Length of output data table wrong'
    return fgt


def fgaintab_from_array(gain: numpy.array, time: numpy.array, antenna: numpy.array, weight: numpy.array,
                        copy=False, meta={}, **kwargs):
    nrows = time.shape[0]
    assert len(antenna) == nrows, "Discrepancy in number of antenna rows"
    assert gain.shape[0] == nrows, "Discrepancy in number of gain rows"
    assert weight.shape[0] == nrows, "Discrepancy in number of weight rows"
    fg = fgaintab()

    fg.data = Table(data=[gain, time, antenna, weight], names=['gain', 'time', 'antenna', 'weight'], copy=copy,
                    meta=meta)
    return fg


if __name__ == '__main__':
    kwargs = {}
    nant = 27
    aantennas = numpy.arange(nant, dtype='int')
    npol = 4
    freq = numpy.arange(5.e7, 15.e7, 2.5e7)
    print(freq)
    atimes = numpy.arange(0.0, 43200.0, 10.0)
    ntimes = len(atimes)
    times = numpy.repeat(atimes, nant)
    antennas = numpy.array(ntimes * list(range(nant)))
    nrows = len(times)
    gains = numpy.ones([len(times), len(freq), npol], dtype='complex')
    weight = numpy.ones([len(times), len(freq)], dtype='float')
    gt = fgaintab_filter(fgaintab_from_array(gains, times, antennas, weight), **kwargs)
    print(gt.data['gain'].shape)
