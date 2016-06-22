# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

import numpy as numpy

from astropy.coordinates import EarthLocation
from astropy.table import Table, Column, vstack
import astropy.units as u

from crocodile.simulate import *


class configuration():
    """
    Describe a configuration
    """
    def __init__(self):
        self.name = ''
        self.data = None
        self.location = None

def configuration_add(fc1: configuration, fc2: configuration):
    """
    Add two configurations together
    :param fc1:
    :param fc2:
    :return:
    """
    fc=configuration()
    fc.name = '%s+%s' % (fc1.name, fc2.name)
    fc.data = vstack(fc1.data, fc2.data)
    fc.location = None


def configuration_filter(fc: configuration, **kwargs):
    print("configuration: No filter implemented yet")
    return fc

def configuration_from_array(antxyz: numpy.array, name: str = None, location: EarthLocation = None,
                       mount: str = 'alt-az', names: str = '%d', meta: dict = None, **kwargs):
    """
    Define from an array
    :rtype: configuration
    :param antxyz: locations of antennas
    :param location: Location of array centre (reference for antenna locations)
    :param mount: Mount type e.g. 'altaz'
    :param names: Generator for names e.g. 'SKA1_MID%d'
    :type meta: dict
    """
    fc = configuration()
    assert len(antxyz) == 2, "Antenna array has wrong shape"
    fc.data = Table(data=[names, antxyz, mount], names=["names", "xyz", "mount"], meta=meta)
    fc.location = location
    return fc


def configuration_from_file(antfile: str, name: str = None, location: EarthLocation = None, mount: str = 'altaz',
                      names: str = "%d", frame: str = 'local', meta: dict = None, **kwargs):
    """
    Define from a file
    :param antfile: Antenna file name
    :param name: Name of array e.g. 'LOWBD2'
    :param location: locationa as EarthLocation
    :param mount: mount type: 'altaz', 'xy'
    :param meta: Any meta info
    """
    fc = configuration()
    fc.name = name
    fc.location = location
    antxyz = numpy.genfromtxt(antfile, delimiter=",")
    assert antxyz.shape[1] == 3, ("Antenna array has wrong shape %s" % antxyz.shape)
    nants = antxyz.shape[0]
    declination = location.geodetic[1].to(u.rad).value
    if frame == 'local':
        rot_xyz = xyz_to_uvw(antxyz, numpy.radians(0), numpy.radians(declination))
        xyz = Column(rot_xyz, name="xyz")
        xyz[:,1], xyz[:,2] = xyz[:,2], xyz[:,1]
        print(rot_xyz)
    else:
        xyz = Column(antxyz, name="xyz")

    anames = [names % (ant) for ant in range(nants)]
    mounts = Column(numpy.repeat(mount, nants), name="mount")
    fc.data = Table(data=[anames, xyz, mounts], names=["names", "xyz", "mount"], meta=meta)
    fc.frame=frame
    return fc

def configuration_from_LOFAR(antfile: str, name: str = None, meta: dict = None, **kwargs):
    """
    :param antfile:
    :param name:
    :param meta:
    :param kwargs:
    :return configuration:
    """
    fc = configuration()
    antxyz = numpy.genfromtxt(antfile, skip_header=2, usecols=[1, 2, 3], delimiter=",")
    nants = antxyz.shape[0]
    assert antxyz.shape[1] == 3, "Antenna array has wrong shape %s" % antxyz.shape
    anames = numpy.genfromtxt(antfile, dtype='str', skip_header=2, usecols=[0], delimiter=",")
    mounts = Column(numpy.repeat('XY', nants), name="mount")
    fc.data = Table(data=[anames, antxyz, mounts], names=["names", "xyz", "mount"], meta=meta)
    fc.location = EarthLocation(x=[3826923.9] * u.m, y=[460915.1] * u.m, z=[5064643.2] * u.m)
    return fc


def named_configuration(name: str = 'LOWBD2', **kwargs):
    """
    Standard configurations e.g. LOWBD2, MIDBD2

    :param name: str name of configuration
    :rtype configuration
    """

    if name == 'LOWBD2':
        # TODO: convert to XYZ
        location = EarthLocation(lon="116.4999", lat="-26.7000", height=300.0)
        fc = configuration_from_file(antfile="./data/configurations/LOWBD2.csv",
                               location=location, mount='xy', names='LOWBD2_%d')
    elif name == 'LOWBD1':
        # TODO: convert to XYZ
        location = EarthLocation(lon="116.4999", lat="-26.7000", height=300.0)
        fc = configuration_from_file(antfile="./data/configurations/LOWBD1.csv",
                               location=location, mount='xy', names='LOWBD1_%d')
    elif name == 'LOFAR':
        fc = configuration_from_LOFAR(antfile="./data/configurations/LOFAR.csv",
                                      frame='geocentric')
    elif name == 'VLAA':
        # TODO: convert to XYZ
        location = EarthLocation(lon="-107.6184", lat="34.0784", height=2124.0)
        fc = configuration_from_file(antfile="./data/configurations/VLA_A_hor_xyz.csv", location=location,
                               mount='altaz',
                               names='VLA_%d')
    else:
        fc = configuration()
        raise UserWarning("No such configuration %s" % name)
    return fc


if __name__ == '__main__':
    import os
    os.chdir('../')
    print(os.getcwd())
    kwargs={}
    fc = configuration()
    for telescope in ['LOWBD1', 'LOWBD2', 'LOFAR', 'VLAA']:
        print(telescope)
        config = configuration_filter(named_configuration(telescope), **kwargs)
        print(config.location)
