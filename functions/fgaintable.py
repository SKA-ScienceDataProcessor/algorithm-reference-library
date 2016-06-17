# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
import numpy as numpy

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.table import Table, Row, Column, MaskedColumn, TableColumns, TableFormatter
from astropy.nddata import NDData

from functions.fcontext import fcontext

class fgaintable(Table):
    """Gain table with time, antenna, frequency
    """
    def __init__(self, gain: numpy.array, time: numpy.array, antenna: numpy.array, weight: numpy.array,
                 names=['gain', 'time', 'antenna', 'weight'], copy=False,
                 meta={'phasecentre':None}, context: fcontext = None, **kwargs):
        nrows = time.shape[0]
        assert len(antenna)==nrows, "Discrepancy in number of rows"
        assert gain.shape[0]==nrows, "Discrepancy in number of rows"
        assert weight.shape[0]==nrows, "Discrepancy in number of rows"
        super.__init__(data=[gain, time, antenna, weight],
        names=['gain', 'time', 'antenna', 'weight'], copy=copy,
        meta=meta)


if __name__ == '__main__':
    import os
    print(os.getcwd())
