"""Configuration definitions

"""

import numpy
from astropy import units as u
from astropy.coordinates import EarthLocation

from processing_library.util.coordinate_support import xyz_at_latitude
from data_models.memory_data_models import Configuration
from data_models.parameters import arl_path, get_parameter
from processing_components.simulation.testing_support import log


def create_configuration_from_file(antfile: str, location: EarthLocation = None,
                                   mount: str = 'altaz',
                                   names: str = "%d", frame: str = 'local',
                                   diameter=35.0,
                                   rmax=None, name='') -> Configuration:
    """ Define from a file

    :param names: Antenna names
    :param antfile: Antenna file name
    :param location:
    :param mount: mount type: 'altaz', 'xy'
    :param frame: 'local' | 'global'
    :param diameter: Effective diameter of station or antenna
    :return: Configuration
    """
    antxyz = numpy.genfromtxt(antfile, delimiter=",")
    assert antxyz.shape[1] == 3, ("Antenna array has wrong shape %s" % antxyz.shape)
    if frame == 'local':
        latitude = location.geodetic[1].to(u.rad).value
        antxyz = xyz_at_latitude(antxyz, latitude)
        antxyz += [location.geocentric[0].to(u.m).value,
                   location.geocentric[1].to(u.m).value,
                   location.geocentric[2].to(u.m).value]
    
    if rmax is not None:
        lantxyz = antxyz - numpy.average(antxyz, axis=0)
        r = numpy.sqrt(lantxyz[:, 0] ** 2 + lantxyz[:, 1] ** 2 + lantxyz[:, 2] ** 2)
        antxyz = antxyz[r < rmax]
        log.debug('create_configuration_from_file: Maximum radius %.1f m includes %d antennas/stations' %
                  (rmax, antxyz.shape[0]))
    else:
        log.debug('create_configuration_from_file: %d antennas/stations' %
                  (antxyz.shape[0]))
    
    nants = antxyz.shape[0]
    anames = [names % ant for ant in range(nants)]
    mounts = numpy.repeat(mount, nants)
    fc = Configuration(location=location, names=anames, mount=mounts, xyz=antxyz, frame=frame,
                       diameter=diameter, name=name)
    return fc


def create_configuration_from_SKAfile(antfile: str,
                                      mount: str = 'altaz',
                                      names: str = "%d",
                                      rmax=None, name='') -> Configuration:
    """ Define from a file

    :param names: Antenna names
    :param antfile: Antenna file name
    :param mount: mount type: 'altaz', 'xy'
    :return: Configuration
    """
    # Diameter, longitude, latitude
    # 15.00	21.44241720	-30.7342510
    # 35	116.7644482	-26.82472208
    antdiamlonglat = numpy.genfromtxt(antfile, skip_header=0, usecols=[0, 1, 2], delimiter="\t")
    assert antdiamlonglat.shape[1] == 3, ("Antenna array has wrong shape %s" % antdiamlonglat.shape)
    antxyz = numpy.zeros([antdiamlonglat.shape[0] - 1, 3])
    diameters = numpy.zeros([antdiamlonglat.shape[0] - 1])
    location = EarthLocation(lon=antdiamlonglat[-1, 1], lat=antdiamlonglat[-1, 2], height=0.0)
    for ant in range(antdiamlonglat.shape[0] - 1):
        loc = EarthLocation(lon=antdiamlonglat[ant, 1], lat=antdiamlonglat[ant, 2], height=0.0).geocentric
        antxyz[ant] = [loc[0].to(u.m).value, loc[1].to(u.m).value, loc[2].to(u.m).value]
        diameters[ant] = antdiamlonglat[ant, 0]
    
    if rmax is not None:
        lantxyz = antxyz - numpy.average(antxyz, axis=0)
        r = numpy.sqrt(lantxyz[:, 0] ** 2 + lantxyz[:, 1] ** 2 + lantxyz[:, 2] ** 2)
        antxyz = antxyz[r < rmax]
        log.debug('create_configuration_from_file: Maximum radius %.1f m includes %d antennas/stations' %
                  (rmax, antxyz.shape[0]))
        diameters = diameters[r < rmax]
    else:
        log.debug('create_configuration_from_file: %d antennas/stations' %
                  (antxyz.shape[0]))
    
    nants = antxyz.shape[0]
    anames = [names % ant for ant in range(nants)]
    mounts = numpy.repeat(mount, nants)
    fc = Configuration(location=location, names=anames, mount=mounts, xyz=antxyz, frame='global',
                       diameter=diameters, name=name)
    return fc


def create_configuration_from_MIDfile(antfile: str, location=None,
                                      mount: str = 'altaz',
                                      rmax=None, name='') -> Configuration:
    """ Define from a file

    :param names: Antenna names
    :param antfile: Antenna file name
    :param mount: mount type: 'altaz', 'xy'
    :return: Configuration
    """
    # X Y Z Diam Station
    # 5109237.714735 2006795.661955 -3239109.183708 13.5 M000
    antdiamlonglat = numpy.genfromtxt(antfile, skip_header=5, usecols=[0, 1, 2, 3, 4], delimiter=" ", dtype="f8, f8, "
                                                                                                            "f8, f8, S8")
    antxyz = numpy.zeros([len(antdiamlonglat), 3])
    diameters = numpy.zeros([len(antdiamlonglat)])
    names = list()
    for ant, line in enumerate(antdiamlonglat):
        lline=list(line)
        antxyz[ant, :] = lline[0:3]
        diameters[ant] = lline[3]
        names.append(lline[4])
    
    if rmax is not None:
        lantxyz = antxyz - numpy.average(antxyz, axis=0)
        r = numpy.sqrt(lantxyz[:, 0] ** 2 + lantxyz[:, 1] ** 2 + lantxyz[:, 2] ** 2)
        antxyz = antxyz[r < rmax]
        log.debug('create_configuration_from_file: Maximum radius %.1f m includes %d antennas/stations' %
                  (rmax, antxyz.shape[0]))
        diameters = diameters[r < rmax]
        names = numpy.array(names)[r < rmax]
    else:
        log.debug('create_configuration_from_file: %d antennas/stations' %
                  (antxyz.shape[0]))
    
    nants = antxyz.shape[0]
    mounts = numpy.repeat(mount, nants)
    fc = Configuration(location=location, names=names, mount=mounts, xyz=antxyz, frame='global',
                       diameter=diameters, name=name)
    return fc


def create_LOFAR_configuration(antfile: str) -> Configuration:
    """ Define from the LOFAR configuration file

    :param antfile:
    :return: Configuration
    """
    antxyz = numpy.genfromtxt(antfile, skip_header=2, usecols=[1, 2, 3], delimiter=",")
    nants = antxyz.shape[0]
    assert antxyz.shape[1] == 3, "Antenna array has wrong shape %s" % antxyz.shape
    anames = numpy.genfromtxt(antfile, dtype='str', skip_header=2, usecols=[0], delimiter=",")
    mounts = numpy.repeat('XY', nants)
    location = EarthLocation(x=[3826923.9] * u.m, y=[460915.1] * u.m, z=[5064643.2] * u.m)
    fc = Configuration(location=location, names=anames, mount=mounts, xyz=antxyz, frame='global',
                       diameter=35.0, name='LOFAR')
    return fc


def create_named_configuration(name: str = 'LOWBD2', **kwargs) -> Configuration:
    """ Standard configurations e.g. LOWBD2, MIDBD2

    :param name: name of Configuration LOWBD2, LOWBD1, LOFAR, VLAA, ASKAP
    :param rmax: Maximum distance of station from the average (m)
    :return:
    
    For LOWBD2, setting rmax gives the following number of stations
    100.0       13
    300.0       94
    1000.0      251
    3000.0      314
    10000.0     398
    30000.0     476
    100000.0    512
    """
    
    if name == 'LOWBD2':
        location = EarthLocation(lon="116.4999", lat="-26.7000", height=300.0)
        fc = create_configuration_from_file(antfile=arl_path("data/configurations/LOWBD2.csv"),
                                            location=location, mount='xy', names='LOWBD2_%d',
                                            diameter=35.0, name=name, **kwargs)
    elif name == 'LOWBD1':
        location = EarthLocation(lon="116.4999", lat="-26.7000", height=300.0)
        fc = create_configuration_from_file(antfile=arl_path("data/configurations/LOWBD1.csv"),
                                            location=location, mount='xy', names='LOWBD1_%d',
                                            diameter=35.0, name=name, **kwargs)
    elif name == 'LOWBD2-CORE':
        location = EarthLocation(lon="116.4999", lat="-26.7000", height=300.0)
        fc = create_configuration_from_file(antfile=arl_path("data/configurations/LOWBD2-CORE.csv"),
                                            location=location, mount='xy', names='LOWBD2_%d',
                                            diameter=35.0, name=name, **kwargs)
    elif name == 'LOWR3':
        fc = create_configuration_from_SKAfile(antfile=arl_path("data/configurations/LOW_SKA-TEL-SKO-0000422_Rev3.txt"),
                                               mount='xy', names='LOWR3_%d',
                                               name=name, **kwargs)
    elif name == 'MID':
        location = EarthLocation(lon="21.443803", lat="-30.712925", height=0.0)
        fc = create_configuration_from_MIDfile(antfile=arl_path("data/configurations/ska1mid.cfg"),
            mount='altaz', name=name, location=location, **kwargs)
    elif name == 'MIDR5':
        fc = create_configuration_from_SKAfile(
            antfile=arl_path("data/configurations/MID_SKA-TEL-INSA-0000537_Rev05.txt"),
            mount='altaz', names='MIDR5_%d',
            name=name, **kwargs)
    elif name == 'ASKAP':
        location = EarthLocation(lon="+116.6356824", lat="-26.7013006", height=377.0)
        fc = create_configuration_from_file(antfile=arl_path("data/configurations/A27CR3P6B.in.csv"),
                                            mount='equatorial', names='ASKAP_%d',
                                            diameter=12.0, name=name, location=location, **kwargs)
    elif name == 'LOFAR':
        assert get_parameter(kwargs, "meta", False) is False
        fc = create_LOFAR_configuration(antfile=arl_path("data/configurations/LOFAR.csv"))
    elif name == 'VLAA':
        location = EarthLocation(lon="-107.6184", lat="34.0784", height=2124.0)
        fc = create_configuration_from_file(antfile=arl_path("data/configurations/VLA_A_hor_xyz.csv"),
                                            location=location,
                                            mount='altaz',
                                            names='VLA_%d',
                                            diameter=25.0, name=name, **kwargs)
    elif name == 'VLAA_north':
        location = EarthLocation(lon="-107.6184", lat="90.000", height=2124.0)
        fc = create_configuration_from_file(antfile=arl_path("data/configurations/VLA_A_hor_xyz.csv"),
                                            location=location,
                                            mount='altaz',
                                            names='VLA_%d',
                                            diameter=25.0, name=name, **kwargs)
    else:
        raise ValueError("No such Configuration %s" % name)
    return fc
