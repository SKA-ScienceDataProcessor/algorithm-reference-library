# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
import numpy as numpy

from astropy.table import Table, Row, Column, MaskedColumn, TableColumns, TableFormatter



class fgaintable(Table):
    """
    Gain table with time, antenna, frequency columns. This is an astropy Table.
    """
    def __init__(self, gain: numpy.array, time: numpy.array, antenna: numpy.array, weight: numpy.array,
                 names=['gain', 'time', 'antenna', 'weight'], copy=False,
                 meta={}, **kwargs):
        nrows = time.shape[0]
        assert len(antenna)==nrows, "Discrepancy in number of antenna rows"
        assert gain.shape[0]==nrows, "Discrepancy in number of gain rows"
        assert weight.shape[0]==nrows, "Discrepancy in number of weight rows"
        super().__init__(data=[gain, time, antenna, weight], names=['gain', 'time', 'antenna', 'weight'],
                         copy=copy, meta=meta)


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
    gt=fgaintable(gains, times, antennas, weight)
    print(gt['gain'].shape)
    gt.show_in_browser()
