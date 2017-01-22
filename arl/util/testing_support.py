# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Definition of structures needed by the function interface. These are mostly
subclasses of astropy classes.
"""

import csv
import unittest

from astropy.coordinates import ICRS, EarthLocation
from astropy.table import Column
from astropy.wcs import WCS

from arl.data.data_models import *
from arl.data.parameters import arl_path
from arl.image.operations import import_image_from_fits, create_image_from_array
from arl.util.coordinate_support import *
from arl.util.read_oskar_vis import OskarVis

log = logging.getLogger(__name__)


def create_configuration_from_array(antxyz: numpy.array, name: str = None, location: EarthLocation = None,
                                    mount: str = 'alt-az', names: str = '%d', meta: dict = None, **kwargs):
    """ Define from parts

    :param params:
    :param name:
    :param antxyz: locations of antennas numpy.array[...,3]
    :param location: Location of array centre (reference for antenna locations)
    :param mount: Mount type e.g. 'altaz'
    :param names: Generator for names e.g. 'SKA1_MID%d'
    :type meta:
    :returns: Configuration
    """
    fc = Configuration()
    assert len(antxyz) == 2, "Antenna array has wrong shape"
    fc.data = Table(data=[names, antxyz, mount], names=["names", "xyz", "mount"], meta=meta)
    fc.location = location
    return fc


def create_configuration_from_file(antfile: str, name: str = None, location: EarthLocation = None, mount: str = 'altaz',
                                   names: str = "%d", frame: str = 'local',
                                   meta: dict = None,
                                   **kwargs):
    """ Define from a file

    :param names:
    :param params:
    :param antfile: Antenna file name
    :param name: Name of array e.g. 'LOWBD2'
    :param location:
    :param mount: mount type: 'altaz', 'xy'
    :param frame: 'local' | 'global'
    :param meta: Any meta info
    :returns: Configuration
    """
    fc = Configuration()
    fc.name = name
    fc.location = location
    antxyz = numpy.genfromtxt(antfile, delimiter=",")
    assert antxyz.shape[1] == 3, ("Antenna array has wrong shape %s" % antxyz.shape)
    nants = antxyz.shape[0]
    if frame == 'local':
        latitude = location.geodetic[1].to(u.rad).value
        antxyz = xyz_at_latitude(antxyz, latitude)
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
    :param meta:
    :param params: Dictionary containing parameters
    :returns: Configuration
    """
    fc = Configuration()
    antxyz = numpy.genfromtxt(antfile, skip_header=2, usecols=[1, 2, 3], delimiter=",")
    nants = antxyz.shape[0]
    assert antxyz.shape[1] == 3, "Antenna array has wrong shape %s" % antxyz.shape
    anames = numpy.genfromtxt(antfile, dtype='str', skip_header=2, usecols=[0], delimiter=",")
    mounts = Column(numpy.repeat('XY', nants), name="mount")
    fc.data = Table(data=[anames, antxyz, mounts], names=["names", "xyz", "mount"], meta=meta)
    fc.location = EarthLocation(x=[3826923.9] * u.m, y=[460915.1] * u.m, z=[5064643.2] * u.m)
    return fc


def create_named_configuration(name: str = 'LOWBD2', **kwargs):
    """ Standard configurations e.g. LOWBD2, MIDBD2

    :param params:
    :param name: name of Configuration LOWBD2, LOWBD1, LOFAR, VLAA
    :returns: Configuration
    """
    
    if name == 'LOWBD2':
        location = EarthLocation(lon="116.4999", lat="-26.7000", height=300.0)
        fc = create_configuration_from_file(antfile=arl_path("data/configurations/LOWBD2.csv"),
                                            location=location, mount='xy', names='LOWBD2_%d')
    elif name == 'LOWBD1':
        location = EarthLocation(lon="116.4999", lat="-26.7000", height=300.0)
        fc = create_configuration_from_file(antfile=arl_path("data/configurations/LOWBD1.csv"),
                                            location=location, mount='xy', names='LOWBD1_%d')
    elif name == 'LOWBD2-CORE':
        location = EarthLocation(lon="116.4999", lat="-26.7000", height=300.0)
        fc = create_configuration_from_file(antfile=arl_path("data/configurations/LOWBD2-CORE.csv"),
                                            location=location, mount='xy', names='LOWBD2_%d')
    elif name == 'LOFAR':
        fc = create_LOFAR_configuration(antfile=arl_path("data/configurations/LOFAR.csv"))
    elif name == 'VLAA':
        location = EarthLocation(lon="-107.6184", lat="34.0784", height=2124.0)
        fc = create_configuration_from_file(antfile=arl_path("data/configurations/VLA_A_hor_xyz.csv"),
                                            location=location,
                                            mount='altaz',
                                            names='VLA_%d')
    elif name == 'VLAA_north':
        location = EarthLocation(lon="-107.6184", lat="90.000", height=2124.0)
        fc = create_configuration_from_file(antfile=arl_path("data/configurations/VLA_A_hor_xyz.csv"),
                                            location=location,
                                            mount='altaz',
                                            names='VLA_%d')
    else:
        fc = Configuration()
        raise UserWarning("No such Configuration %s" % name)
    return fc


def import_visibility_from_oskar(oskar_file: str, **kwargs) -> Visibility:
    """ Import a visibility set from an OSKAR visibility file

    :param params:
    :param oskar_file: Name of OSKAR visibility file
    :returns: Visibility
    """
    
    # Extract data from Oskar file
    oskar_vis = OskarVis(oskar_file)
    ra, dec = oskar_vis.phase_centre()
    a1, a2 = oskar_vis.stations(flatten=True)
    
    # Make configuration
    location = EarthLocation(lon=oskar_vis.telescope_lon,
                             lat=oskar_vis.telescope_lat,
                             height=oskar_vis.telescope_alt)
    antxyz = numpy.transpose([oskar_vis.station_x,
                              oskar_vis.station_y,
                              oskar_vis.station_z])
    config = Configuration(
        name=oskar_vis.telescope_path,
        location=location,
        xyz=antxyz
    )
    
    # Construct visibilities
    return Visibility(
        frequency=[oskar_vis.frequency(i) for i in range(oskar_vis.num_channels)],
        phasecentre=SkyCoord(frame=ICRS, ra=ra, dec=dec, unit=u.deg),
        configuration=config,
        uvw=numpy.transpose(oskar_vis.uvw(flatten=True)),
        time=oskar_vis.times(flatten=True),
        antenna1=a1,
        antenna2=a2,
        vis=oskar_vis.amplitudes(flatten=True),
        weight=numpy.ones(a1.shape))


def create_test_image(canonical=True, npol=4, nchan=1, cellsize=None):
    """Create a useful test image

    This is the test image M31 widely used in ALMA and other simulations. It is actually part of an Halpha region in
    M31.

    :param canonical: Make the image into a 4 dimensional image
    :param npol: Number of polarisations
    :param nchan: Number of channels
    :returns: Image
    """
    im = import_image_from_fits(arl_path("data/models/M31.MOD"))
    if canonical:
        im = replicate_image(im, nchan=nchan, npol=npol)
        if cellsize is not None:
            im.wcs.wcs.cdelt[0] = -180.0 * cellsize / numpy.pi
            im.wcs.wcs.cdelt[1] = +180.0 * cellsize / numpy.pi
        im.wcs.wcs.radesys = 'ICRS'
        im.wcs.wcs.equinox = 2000.00
    return im


def create_low_test_image(npixel=16384, npol=1, nchan=1, cellsize=0.000015, frequency=1e8, channelwidth=1e6,
                          phasecentre=None):
    """Create LOW test image from S3
    
    The input catalog was generated at http://s-cubed.physics.ox.ac.uk/s3_sex using the following query
    Database: s3_sex
    SQL: select * from Galaxies where (pow(10,itot_151)*1000 > 1.0) and (right_ascension between -5 and 5) and (declination between -5 and 5);;
    Number of rows returned: 29966
    
    :param npixel: Number of pixels
    :param npol: Number of polarisations (all set to same value)
    :param nchan: Number of channels (all set to same value)
    :param cellsize: cellsize in radians
    :param reffrequency: Reference frequency (Hz)
    :param channelwidth: Channel width (Hz)
    :param phasecentre: phasecentre (SkyCoord)
    :returns: Image
    """
    
    ras = []
    decs = []
    fluxes = []
    
    if phasecentre is None:
        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox=2000.0)
    
    shape = [nchan, npol, npixel, npixel]
    w = WCS(naxis=4)
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, 1.0, channelwidth]
    w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.value, phasecentre.dec.value, 1.0, frequency]
    w.naxis = 4
    
    w.wcs.radesys = 'ICRS'
    w.wcs.equinox = 2000.0
    
    model = create_image_from_array(numpy.zeros(shape), w)
    
    with open(arl_path('data/models/S3_151MHz_10deg.csv')) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        r = 0
        for row in readCSV:
            # Skip first row
            if r > 0:
                ra = float(row[4]) + 180.0
                dec = float(row[5]) - 60.0
                flux = numpy.power(10, float(row[9]))
                ras.append(ra)
                decs.append(dec)
                fluxes.append(flux)
            r += 1
    
    p = w.sub(2).wcs_world2pix(numpy.array(ras), numpy.array(decs), 0)
    fluxes = numpy.array(fluxes)
    ip = numpy.round(p).astype('int')
    ok = numpy.where((0 <= ip[0,:]) & (npixel > ip[0,:]) & (0 <= ip[1,:]) & (npixel > ip[1,:]))[0]
    ps = ip[:,ok]
    log.info('create_low_test_image: %d sources inside the image' % (ps.shape[1]))
    for chan in range(nchan):
        for pol in range(npol):
            model.data[chan, pol, ip[1,ok], ip[0,ok]] += fluxes[ok]
    
    return model


def replicate_image(im: Image, npol=4, nchan=1, frequency=1.4e9):
    """ Make a new canonical shape Image, extended along third and fourth axes by replication.

    The order of the data is [chan, pol, dec, ra]


    :param frequency:
    :param im:
    :param npol: Number of polarisation axes
    :param nchan: Number of spectral channels
    :returns: Image
    """
    
    if len(im.data.shape) == 2:
        fim = Image()
        
        newwcs = WCS(naxis=4)
        
        newwcs.wcs.crpix = [im.wcs.wcs.crpix[0], im.wcs.wcs.crpix[1], 1.0, 1.0]
        newwcs.wcs.cdelt = [im.wcs.wcs.cdelt[0], im.wcs.wcs.cdelt[1], 1.0, 1.0]
        newwcs.wcs.crval = [im.wcs.wcs.crval[0], im.wcs.wcs.crval[1], 1.0, frequency]
        newwcs.wcs.ctype = [im.wcs.wcs.ctype[0], im.wcs.wcs.ctype[1], 'STOKES', 'FREQ']
        
        fim.wcs = newwcs
        fshape = [nchan, npol, im.data.shape[1], im.data.shape[0]]
        fim.data = numpy.zeros(fshape)
        log.info("replicate_image: replicating shape %s to %s" % (im.data.shape, fim.data.shape))
        for i3 in range(nchan):
            for i2 in range(npol):
                fim.data[i3, i2, :, :] = im.data[:, :]
    else:
        fim = im
    
    return fim


def run_unittests(logLevel=logging.DEBUG, *args, **kwargs):
    """Runs the unit tests in all loaded modules.

    :param logLevel: The amount of logging to generate. By default, we
      show all log messages (level DEBUG)
    :param *args: Will be passed to `unittest.main`
    """
    
    # Set up logging environment
    rootLog = logging.getLogger()
    rootLog.setLevel(logLevel)
    rootLog.addHandler(logging.StreamHandler(sys.stderr))
    
    # Call unittest main
    unittest.main(*args, **kwargs)
    
if __name__ == '__main__':
    im=create_low_test_image()

