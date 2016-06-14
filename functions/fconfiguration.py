# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
import numpy as numpy

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.table import Table, Row, Column, MaskedColumn, TableColumns, TableFormatter
from astropy.nddata import NDData
import astropy.units as u

from crocodile.simulate import *

class fconfiguration(Table):
    """
    Describe a configuration
    """
    def __init__(self, antxyz: numpy.array, location: object = None,
                 mount: object = 'alt-az', names: object = '%d', meta: object = {}) -> object:
        """
        Define from an array
        :param antxyz: locations of antennas
        :param location: Location of array centre (reference for antenna locations)
        :param mount: Mount type e.g. 'altaz'
        :param names: Generator for names e.g. 'SKA1_MID%d'
        :type meta: dict
        """
        assert len(antxyz)==2, "Antenna array has wrong shape"
        super().__init__(data=[names, NDData(self.antxyz), mount], names=["names", "xyz", "mount"], meta=meta)

    def __init__(self, antfile: str, location: EarthLocation=None, mount: str='altaz', names: str="%d",
                 meta: dict={}):
        """
        Define from a file
        :type names: vistable
        :param antfile:
        :param location:
        :param mount:
        :param meta:
        """
        meta['location']=location
        antxyz=numpy.genfromtxt(antfile, delimiter=",")
        assert antxyz.shape[1]==3, "Antenna array has wrong shape"

        nants=antxyz.shape[0]
        rot_xyz=xyz_at_latitude(antxyz, location.geodetic[1].to(u.rad).value)
        xyz=Column(rot_xyz, name="xyz")
        anames=[names % (ant) for ant in range(nants)]
        mounts=Column(numpy.repeat(mount, nants), name="mounts")
        assert antxyz.shape[1] == 3, "Antenna array has wrong shape %s" % antxyz.shape
        super().__init__(data=[anames, xyz, mounts, ], names=["mount", "xyz", "names"], meta=meta)

if __name__ == '__main__':
    import os
    print(os.getcwd())
    vlalocation=EarthLocation.from_geodetic(lat="34.0784", lon="107.6184")
    config = fconfiguration("../data/vis/VLA_A_hor_xyz.txt", location=vlalocation,
                            names='VLA_%d')
    print(config)