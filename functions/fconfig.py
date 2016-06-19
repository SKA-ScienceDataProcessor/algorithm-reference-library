# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

from collections import namedtuple

import numpy as numpy

import re

from astropy.coordinates import EarthLocation
from astropy.table import Table, Row, Column, MaskedColumn, TableColumns, TableFormatter
import astropy.units as u

from crocodile.simulate import *


def fconfig():
    """
    Describe a configuration
    Columns: name, antxyz, mount,
    """
    return namedtuple('fconfig', ['name', 'data', 'location'])


def fconfig_from_array(antxyz: numpy.array, name: str = None, location: EarthLocation = None,
                       mount: str = 'alt-az', names: str = '%d', meta: dict = None, **kwargs):
    """
    Define from an array
    :rtype: fconfig
    :param antxyz: locations of antennas
    :param location: Location of array centre (reference for antenna locations)
    :param mount: Mount type e.g. 'altaz'
    :param names: Generator for names e.g. 'SKA1_MID%d'
    :type meta: dict
    """
    fc = fconfig()
    assert len(antxyz) == 2, "Antenna array has wrong shape"
    fc.data = Table(data=[names, numpy.array(self.antxyz), mount], names=["names", "xyz", "mount"], meta=meta)
    fc.location = location
    return fc


def fconfig_from_file(antfile: str, name: str = None, location: EarthLocation = None, mount: str = 'altaz',
                      names: str = "%d", meta: dict = None, **kwargs):
    """
    Define from a file
    :param antfile: Antenna file name
    :param name: Name of array e.g. 'LOWBD2'
    :param location: locationa as EarthLocation
    :param mount: mount type: 'altaz', 'xy'
    :param meta: Any meta info
    """
    fc = fconfig()
    fc.name = name
    fc.location = location
    antxyz = numpy.genfromtxt(antfile, delimiter=",")
    assert antxyz.shape[1] == 3, ("Antenna array has wrong shape %s" % antxyz.shape)
    nants = antxyz.shape[0]
    rot_xyz = xyz_at_latitude(antxyz, location.geodetic[1].to(u.rad).value)
    xyz = Column(rot_xyz, name="xyz")
    anames = [names % (ant) for ant in range(nants)]
    mounts = Column(numpy.repeat(mount, nants), name="mount")
    fc.data = Table(data=[anames, xyz, mounts], names=["names", "xyz", "mount"], meta=meta)
    return fc


def fconfig_from_LOFAR(antfile: str, name: str = None, meta: dict = None, **kwargs):
    """
    :param antfile:
    :param name:
    :param meta:
    :param kwargs:
    :return fconfig:
    """
    fc= fconfig()
    antxyz = numpy.genfromtxt(antfile, skip_header=2, usecols=[1, 2, 3], delimiter=",")
    nants = antxyz.shape[0]
    assert antxyz.shape[1] == 3, "Antenna array has wrong shape %s" % antxyz.shape
    anames = numpy.genfromtxt(antfile, dtype='str', skip_header=2, usecols=[0], delimiter=",")
    mounts = Column(numpy.repeat('XY', nants), name="mount")
    fc.data = Table(data=[anames, antxyz, mounts], names=["names", "xyz", "mount"], meta=meta)
    fc.location = EarthLocation(x=[3826923.9] * u.m, y=[460915.1] * u.m, z=[5064643.2] * u.m)
    return fc


def fconfig_from_name(name: str = 'LOWBD2', **kwargs):
    """
    Standard configurations e.g. LOWBD2, MIDBD2

    :param name: str name of configuration
    :rtype fconfig
    """

    if name == 'LOWBD2':
        location = EarthLocation(lon="116.4999", lat="-26.7000", height=300.0)
        fc = fconfig_from_file(antfile="../data/configurations/LOWBD2.csv",
                               location=location, mount='xy', names='LOWBD2_%d')
    elif name == 'LOWBD1':
        location = EarthLocation(lon="116.4999", lat="-26.7000", height=300.0)
        fc = fconfig_from_file(antfile="../data/configurations/LOWBD1.csv",
                               location=location, mount='xy', names='LOWBD1_%d')
    elif name == 'LOFAR':
        fc = fconfig_from_LOFAR(antfile="../data/configurations/LOFAR.csv")
    elif name == 'VLAA':
        location = EarthLocation(lon="-107.6184", lat="34.0784", height=2124.0)
        fc = fconfig_from_file(antfile="../data/configurations/VLA_A_hor_xyz.csv", location=location,
                 mount='altaz',
                 names='VLA_%d')
    else:
        fc=fconfig()
        raise UserWarning("No such configuration %s" % name)
    return fc

if __name__ == '__main__':
    fc = fconfig()
    for telescope in ['LOWBD1', 'LOWBD2', 'LOFAR', 'VLAA']:
        print(telescope)
        config=fconfig_from_name(telescope)
        print(config.location)
