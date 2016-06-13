# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
import numpy as numpy

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.table import Table, Row, Column, MaskedColumn, TableColumns, TableFormatter
from astropy.wcs import WCS
from astropy.nddata import NDData

from fconfiguration import fconfiguration

class fvistable(Table):
    """Visibility class with uvw, time, a1, a2, vis, weight columns
    """
    def __init__(self, uvw: NDData, time: NDData, antenna1: NDData, antenna2: NDData,
                 vis: NDData, weight: NDData, copy=False,
                 meta={'phasecentre':None}):
        """
        Create a vistable from the individual parts
        :param uvw:
        :param time:
        :param antenna1:
        :param antenna2:
        :param vis:
        :param weight:
        :param copy:
        :param meta:
        """
        nrows = time.shape[0]
        assert uvw.shape[0]==nrows, "Discrepancy in number of rows"
        assert len(antenna1)==nrows, "Discrepancy in number of rows"
        assert len(antenna2)==nrows, "Discrepancy in number of rows"
        assert vis.shape[0]==nrows, "Discrepancy in number of rows"
        assert weight.shape[0]==nrows, "Discrepancy in number of rows"
        super(fvistable, self).__init__(data=[uvw, time, antenna1, antenna2, vis, weight],
                                        names=['uvw', 'time', 'antenna1', 'antenna2', 'vis', 'weight'],
                                        copy=copy, meta=meta)
    def __init__(self, config: fconfiguration, times: numpy.array, weight: float, direction: SkyCoord=None):
        """
        Creat a vistable from configuration, hour angles, and direction of source
        :param config: Configuration of antennas
        :param times: hour angles
        :param weight: weight of a single sample
        :param direction: direction of source
        """


if __name__ == '__main__':
    import os
    print(os.getcwd())
    config = fconfiguration("../data/vis/VLA_A_hor_xyz.txt", location=EarthLocation.of_site('mro'))
    times=numpy.pi*numpy.arange(-6.0,+6.0,0.1)/12.0
    mrolocation=EarthLocation.of_site('mro')
    vt=fvistable(config, times, weight=1.0)