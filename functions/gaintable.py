# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
import numpy as numpy

from astropy.table import Table, vstack

"""
Functions that define and manipulate gain tables
"""

class GainTable():
    """
    Gain table with time, antenna, gain[:,chan,pol] columns
    """
    # TODO: Implement gaintables with Jones and Mueller matrices

    def __init__(self):
        self.data = None
        self.frequency = None


def gaintable_filter(fg: GainTable, **kwargs):
    """

    :param fg:
    :param kwargs:
    :return:
    """
    print("GainTable: Filter not implemented yet")
    return fg


def gaintable_add(fgt1: GainTable, fgt2: GainTable, **kwargs):
    """
    Add two gaintables
    :param fgt1:
    :param fgt2:
    :param kwargs:
    :return:
    """
    assert len(fgt1.frequency) == len(fgt2.frequency), "GainTable: frequencies should be the same"
    assert numpy.max(numpy.abs(fgt1.frequency - fgt2.frequency)) < 1.0, "GainTable: frequencies should be the same"
    print("GainTable: adding tables with %d rows and %d rows" % (len(fgt1.data), len(fgt2.data)))
    fgt = GainTable()
    fgt.data = vstack([fgt1.data, fgt2.data], join_type='exact')
    print(u"Created table with {0:d} rows".format(len(fgt.data)))
    assert (len(fgt.data) == (len(fgt1.data) + len(fgt2.data))), 'Length of output data table wrong'
    return fgt


def gaintable_from_array(gain: numpy.array, time: numpy.array, antenna: numpy.array, weight: numpy.array,
                         frequency: numpy.array, copy=False, meta={}, **kwargs):
    """
    Create a gaintable from arrays
    :param gain:
    :param time:
    :param antenna:
    :param weight:
    :param frequency:
    :param copy:
    :param meta:
    :param kwargs:
    :return:
    """
    nrows = time.shape[0]
    assert len(frequency) == gain.shape[1], "Discrepancy in frequency channels"
    assert len(antenna) == nrows, "Discrepancy in number of antenna rows"
    assert gain.shape[0] == nrows, "Discrepancy in number of gain rows"
    assert weight.shape[0] == nrows, "Discrepancy in number of weight rows"
    fg = GainTable()

    fg.data = Table(data=[gain, time, antenna, weight], names=['gain', 'time', 'antenna', 'weight'], copy=copy,
                    meta=meta)
    fg.frequency = frequency
    return fg


if __name__ == '__main__':
    import os

    os.chdir('../')
    print(os.getcwd())
    kwargs = {}
    nant = 27
    aantennas = numpy.arange(nant, dtype='int')
    npol = 4
    freq = numpy.arange(5.e7, 15.e7, 2e7)
    print(freq)
    atimes = numpy.arange(-43200.0, 43200.0, 30.0)
    ntimes = len(atimes)
    times = numpy.repeat(atimes, nant)
    antennas = numpy.array(ntimes * list(range(nant)))
    nrows = len(times)
    gains = numpy.ones([len(times), len(freq), npol], dtype='complex')
    weight = numpy.ones([len(times), len(freq)], dtype='float')
    gt = gaintable_filter(gaintable_from_array(gains, times, antennas, weight, freq), **kwargs)
    print(gt.data['gain'].shape)
