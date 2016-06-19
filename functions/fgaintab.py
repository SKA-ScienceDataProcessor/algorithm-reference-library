# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
from collections import namedtuple
import numpy as numpy

from astropy.table import Table, Row, Column, MaskedColumn, TableColumns, TableFormatter


def fgaintab():
    """
    Gain table with time, antenna, frequency columns.
    """
    return namedtuple('fgaintab', ['data'])

def fgaintab_from_array(gain: numpy.array, time: numpy.array, antenna: numpy.array, weight: numpy.array,
                        copy=False, meta={}, **kwargs):
    nrows = time.shape[0]
    assert len(antenna)==nrows, "Discrepancy in number of antenna rows"
    assert gain.shape[0]==nrows, "Discrepancy in number of gain rows"
    assert weight.shape[0]==nrows, "Discrepancy in number of weight rows"
    fg=fgaintab()

    fg.data=Table(data=[gain, time, antenna, weight], names=['gain', 'time', 'antenna', 'weight'], copy=copy, meta=meta)
    return fg


if __name__ == '__main__':
    nant=27
    aantennas=numpy.arange(nant,dtype='int')
    npol=4
    freq=numpy.arange(5.e7,15.e7,2.5e7)
    print(freq)
    atimes=numpy.arange(0.0,43200.0,10.0)
    ntimes=len(atimes)
    times=numpy.repeat(atimes, nant)
    antennas=numpy.array(ntimes*list(range(nant)))
    nrows=len(times)
    gains=numpy.ones([len(times), len(freq), npol], dtype='complex')
    weight=numpy.ones([len(times), len(freq)], dtype='float')
    gt=fgaintab_from_array(gains, times, antennas, weight)
    print(gt.data['gain'].shape)
