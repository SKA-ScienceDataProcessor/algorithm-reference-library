# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
import numpy as numpy

import re

from astropy.coordinates import EarthLocation
from astropy.table import Table, Row, Column, MaskedColumn, TableColumns, TableFormatter
from astropy.nddata import NDData
import astropy.units as u

from crocodile.simulate import *

class fconfiguration(Table):
    """
    Describe a configuration
    Columns: name, antxyz, mount,
    """
    def __init__(self):
        self.location=EarthLocation(lon=0.0, lat=0.0)
        pass

    def fromarray(self, antxyz: NDData, location: EarthLocation = None,
                 mount: str = 'alt-az', names: str = '%d', meta: dict = {}) -> object:
        """
        Define from an array
        :rtype: object
        :param antxyz: locations of antennas
        :param location: Location of array centre (reference for antenna locations)
        :param mount: Mount type e.g. 'altaz'
        :param names: Generator for names e.g. 'SKA1_MID%d'
        :type meta: dict
        """
        assert len(antxyz)==2, "Antenna array has wrong shape"
        super().__init__(data=[names, NDData(self.antxyz), mount], names=["names", "xyz", "mount"], meta=meta)
        self.location=location
        return self

    def fromfile(self, antfile: str, location: EarthLocation=None, mount: str='altaz', names: str="%d",
                 meta: dict={}) -> object:
        """
        Define from a file
        :param antfile: Antenna file name
        :param location: locationa as EarthLocation
        :param mount: mount type: 'altaz', 'xy'
        :param meta: Any meta info
        """
        antxyz=numpy.genfromtxt(antfile, delimiter=",")
        assert antxyz.shape[1] == 3, "Antenna array has wrong shape %s" % antxyz.shape

        nants=antxyz.shape[0]
        rot_xyz=xyz_at_latitude(antxyz, location.geodetic[1].to(u.rad).value)
        xyz=Column(rot_xyz, name="xyz")
        anames=[names % (ant) for ant in range(nants)]
        mounts=Column(numpy.repeat(mount, nants), name="mounts")
        super().__init__(data=[anames, xyz, mounts], names=["names", "xyz", "mount"], meta=meta)
        self.location=location
        return self

    def __fromLOFAR(self, antfile: str, meta: dict = {}) -> object:
        antxyz = numpy.genfromtxt(antfile, skip_header=2, usecols=[1,2,3], delimiter=",")
        nants=antxyz.shape[0]
        assert antxyz.shape[1] == 3, "Antenna array has wrong shape %s" % antxyz.shape
        anames = numpy.genfromtxt(antfile, dtype='str', skip_header=2, usecols=[0], delimiter=",")
        mounts = Column(numpy.repeat('XY', nants), name="mounts")
        super().__init__(data=[anames, antxyz, mounts], names=["names", "xyz", "mount"], meta=meta)
        self.location = EarthLocation(x=[3826923.9] * u.m, y=[460915.1] * u.m, z=[5064643.2] * u.m)
        return self

    def fromname(self, name : str ='LOWBD2') -> object:
        """
        Standard configurations e.g. LOWBD2, MIDBD2

        :param name: str name of configuration
        :rtype object
        """
        if name == 'LOWBD2':
            location = EarthLocation(lon="116.4999", lat="-26.7000", height=300.0)
            fconfiguration.fromfile(self, antfile="../data/configurations/LOWBD2.csv",
                                    location=location,
                                    mount='xy',
                                    names='LOWBD2_%d')
        elif name == 'LOWBD1':
            location=EarthLocation(lon="116.4999",lat= "-26.7000", height=300.0)
            fconfiguration.fromfile(self, antfile="../data/configurations/LOWBD1.csv",
                                    location=location,
                                    mount='xy', names='LOWBD1_%d')
        elif name == 'LOFAR':
            fconfiguration.__fromLOFAR(self, antfile="../data/configurations/LOFAR.csv")
        elif name == 'VLAA':
            location = EarthLocation(lon="107.6184", lat="34.0784", height=2124.0)
            fconfiguration.fromfile(self, antfile="../data/configurations/VLA_A_hor_xyz.csv",
                                    location=location,
                                    mount='xy',
                                    names='VLA_%d')
        else:
            raise UserWarning("No such configuration %s" % name)
        return self

    def known(self):
        return ['LOWBD2', 'LOWBD1', 'VLAA', 'LOFAR']

if __name__ == '__main__':
    import os
    print(os.getcwd())

    for telescope in fconfiguration().known():
        config = fconfiguration()
        config.fromname(telescope)
        print(config.location.to_geodetic())
        print(config['names'])
        config['names'][4]='New name'
        print(config['names'][0:10])
        print(config)
