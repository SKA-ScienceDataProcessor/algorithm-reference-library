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

class fmkernel(Table):
    """ Mueller kernel with NDData, antenna1, antenna2, time
    """

class fjones(Table):
    """ Jones kernel with NDData, antenna1, antenna2, time
    """

if __name__ == '__main__':
    import os
    print(os.getcwd())
