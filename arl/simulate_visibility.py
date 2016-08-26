# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
import numpy as numpy

from astropy.table import Table

import os

import numpy

import astropy.units as units
from astropy.coordinates import EarthLocation
from astropy.table import Table, Column, vstack

from crocodile.simulate import *

from arl.data_models import *

"""
Functions that define and manipulate telescope configurations. These are required for simulations.
"""

def filter_configuration(fc: Configuration, **kwargs):
    """ Filter a configuration e.g. remove certain antennas

    :param fc:
    :type Configuration:
    :param kwargs:
    :returns: Configuration
    """
    print("filter_configuration: No filter implemented yet")
    return fc


def create_configuration_from_array(antxyz: numpy.array, name: str = None, location: EarthLocation = None,
                                    mount: str = 'alt-az', names: str = '%d', meta: dict = None, **kwargs):
    """ Define from parts

    :param name:
    :param antxyz: locations of antennas numpy.array[...,3]
    :type numpy.array:
    :param location: Location of array centre (reference for antenna locations)
    :type EarthLocation:
    :param mount: Mount type e.g. 'altaz'
    :type str:
    :param names: Generator for names e.g. 'SKA1_MID%d'
    :type generator:
    :type meta:
    :type dict:
    :returns: Configuration
    """
    fc = Configuration()
    assert len(antxyz) == 2, "Antenna array has wrong shape"
    fc.data = Table(data=[names, antxyz, mount], names=["names", "xyz", "mount"], meta=meta)
    fc.location = location
    return fc


def create_configuration_from_file(antfile: str, name: str = None, location: EarthLocation = None, mount: str = 'altaz',
                                   names: str = "%d", frame: str = 'local',
                                   meta: dict =
                                   None,
                                   **kwargs):
    """ Define from a file

    :param antfile: Antenna file name
    :type str:
    :param name: Name of array e.g. 'LOWBD2'
    :type str:
    :param location:
    :type EarthLocation:
    :param mount: mount type: 'altaz', 'xy'
    :type str:
    :param frame: 'local' | 'global'
    :type str:
    :param meta: Any meta info
    :type dict:
    :returns: Configuration
    """
    fc = Configuration()
    fc.name = name
    fc.location = location
    antxyz = numpy.genfromtxt(antfile, delimiter=",")
    assert antxyz.shape[1] == 3, ("Antenna array has wrong shape %s" % antxyz.shape)
    nants = antxyz.shape[0]
    declination = location.geodetic[1].to(units.rad).value
    if frame == 'local':
        rot_xyz = xyz_to_uvw(antxyz, numpy.zeros(1), numpy.radians(declination))
        xyz = Column(rot_xyz, name="xyz")
        xyz[:, 1], xyz[:, 2] = xyz[:, 2], xyz[:, 1]
    else:
        xyz = Column(antxyz, name="xyz")
    
    anames = [names % ant for ant in range(nants)]
    mounts = Column(numpy.repeat(mount, nants), name="mount")
    fc.data = Table(data=[anames, xyz, mounts], names=["names", "xyz", "mount"], meta=meta)
    fc.frame = frame
    return fc


def create_LOFAR_configuration(antfile: str, meta: dict = None,
                               **kwargs):
    """ Define from the LOFAR configuration file

    :param antfile:
    :type str:
    :param name:
    :type str:
    :param meta:
    :type dict:
    :param kwargs:
    :returns: Configuration
    """
    fc = Configuration()
    antxyz = numpy.genfromtxt(antfile, skip_header=2, usecols=[1, 2, 3], delimiter=",")
    nants = antxyz.shape[0]
    assert antxyz.shape[1] == 3, "Antenna array has wrong shape %s" % antxyz.shape
    anames = numpy.genfromtxt(antfile, dtype='str', skip_header=2, usecols=[0], delimiter=",")
    mounts = Column(numpy.repeat('XY', nants), name="mount")
    fc.data = Table(data=[anames, antxyz, mounts], names=["names", "xyz", "mount"], meta=meta)
    fc.location = EarthLocation(x=[3826923.9] * units.m, y=[460915.1] * units.m, z=[5064643.2] * units.m)
    return fc


def create_named_configuration(name: str = 'LOWBD2', **kwargs):
    """ Standard configurations e.g. LOWBD2, MIDBD2

    :param name: name of Configuration LOWBD2, LOWBD1, LOFAR, VLAA
    :type str:
    :returns: Configuration
    """
    
    chome = os.getenv('CROCODILE', './')
    
    if name == 'LOWBD2':
        location = EarthLocation(lon="116.4999", lat="-26.7000", height=300.0)
        fc = create_configuration_from_file(antfile="%s/data/configurations/LOWBD2.csv" % chome,
                                            location=location, mount='xy', names='LOWBD2_%d')
    elif name == 'LOWBD1':
        location = EarthLocation(lon="116.4999", lat="-26.7000", height=300.0)
        fc = create_configuration_from_file(antfile="%s/data/configurations/LOWBD1.csv" % chome,
                                            location=location, mount='xy', names='LOWBD1_%d')
    elif name == 'LOFAR':
        fc = create_LOFAR_configuration(antfile="%s/data/configurations/LOFAR.csv" % chome,
                                        frame='geocentric')
    elif name == 'VLAA':
        location = EarthLocation(lon="-107.6184", lat="34.0784", height=2124.0)
        fc = create_configuration_from_file(antfile="%s/data/configurations/VLA_A_hor_xyz.csv" % chome,
                                            location=location,
                                            mount='altaz',
                                            names='VLA_%d')
    else:
        fc = Configuration()
        raise UserWarning("No such Configuration %s" % name)
    return fc

if __name__ == '__main__':
    import os
    from arl.define_visibility import create_gaintable_from_array, filter_gaintable
    
    os.chdir('../')
    print(os.getcwd())

    kwargs = {}
    fc = Configuration()
    for telescope in ['LOWBD1', 'LOWBD2', 'LOFAR', 'VLAA']:
        print(telescope)
        config = filter_configuration(create_named_configuration(telescope), **kwargs)
        print(config.location)
        
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
    gt = filter_gaintable(create_gaintable_from_array(gains, times, antennas, weight, freq), **kwargs)
    print(gt.data['gain'].shape)
