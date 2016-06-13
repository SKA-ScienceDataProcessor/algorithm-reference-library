# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
import numpy as numpy

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.table import Table, Row, Column, MaskedColumn, TableColumns, TableFormatter
from astropy.nddata import NDData

class fconfiguration(Table):
    """ Describe a configuration
    """
    def __init__(self, antxyz: numpy.array, location: EarthLocation=None, mount: str='alt-az', meta: dict ={}):
        """ Define from an array
        param: antxyz: locations of antennas
        :type meta: dict
        """
        assert len(antxyz)==2, "Antenna array has wrong shape"
        super().__init__(data=[NDData(self.antxyz, mount)], names=["xyz", "mount"], meta=meta)

    def __init__(self, antfile: str, location: EarthLocation=None, mount: str='altaz', meta: dict={}):
        antxyz=numpy.genfromtxt(antfile, delimiter=",")
        nants=antxyz.shape[0]
        xyz=Column(antxyz, name="xyz")
        mounts=Column(numpy.repeat(mount, nants), name="mounts")
        assert antxyz.shape[1] == 3, "Antenna array has wrong shape %s" % antxyz.shape
        super().__init__(data=[xyz, mounts], meta=meta)

if __name__ == '__main__':
    import os
    print(os.getcwd())
    config = fconfiguration("../data/vis/VLA_A_hor_xyz.txt", location=EarthLocation.of_site('mro'))
    print("The MRO is located at %s" % EarthLocation.of_site('mro'))
    print(config)