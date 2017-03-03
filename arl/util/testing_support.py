# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Definition of structures needed by the function interface.
"""

import csv

import numpy

from reproject import reproject_interp

from astropy.coordinates import EarthLocation
from astropy.wcs import WCS

from arl.calibration.gaintable import *
from arl.data.parameters import arl_path
from arl.fourier_transforms.ftprocessor_base import predict_2d, predict_skycomponent_blockvisibility
from arl.image.operations import import_image_from_fits, create_image_from_array, \
    reproject_image, create_empty_image_like
from arl.util.coordinate_support import *
from arl.visibility.coalesce import coalesce_visibility
from arl.visibility.operations import create_blockvisibility, copy_visibility

log = logging.getLogger(__name__)


def create_configuration_from_file(antfile: str, name: str = None, location: EarthLocation = None,
                                   mount: str = 'altaz',
                                   names: str = "%d", frame: str = 'local',
                                   meta: dict = None,
                                   **kwargs):
    """ Define from a file

    :param names:
    :param antfile: Antenna file name
    :param name: Name of array e.g. 'LOWBD2'
    :param location:
    :param mount: mount type: 'altaz', 'xy'
    :param frame: 'local' | 'global'
    :param meta: Any meta info
    :returns: Configuration
    """
    antxyz = numpy.genfromtxt(antfile, delimiter=",")
    assert antxyz.shape[1] == 3, ("Antenna array has wrong shape %s" % antxyz.shape)
    nants = antxyz.shape[0]
    if frame == 'local':
        latitude = location.geodetic[1].to(u.rad).value
        antxyz = xyz_at_latitude(antxyz, latitude)
    anames = [names % ant for ant in range(nants)]
    mounts = numpy.repeat(mount, nants)
    fc = Configuration(location=location, names=anames, mount=mounts, xyz=antxyz, frame=frame)
    return fc


# noinspection PyUnresolvedReferences,PyUnresolvedReferences,PyUnresolvedReferences
def create_LOFAR_configuration(antfile: str, meta: dict = None,
                               **kwargs):
    """ Define from the LOFAR configuration file

    :param antfile:
    :param meta:
    :returns: Configuration
    """
    antxyz = numpy.genfromtxt(antfile, skip_header=2, usecols=[1, 2, 3], delimiter=",")
    nants = antxyz.shape[0]
    assert antxyz.shape[1] == 3, "Antenna array has wrong shape %s" % antxyz.shape
    anames = numpy.genfromtxt(antfile, dtype='str', skip_header=2, usecols=[0], delimiter=",")
    mounts = numpy.repeat('XY', nants)
    location = EarthLocation(x=[3826923.9] * u.m, y=[460915.1] * u.m, z=[5064643.2] * u.m)
    fc = Configuration(location=location, names=anames, mount=mounts, xyz=antxyz, frame='global')
    return fc


def create_named_configuration(name: str = 'LOWBD2', **kwargs):
    """ Standard configurations e.g. LOWBD2, MIDBD2

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


def create_test_image(canonical=True, cellsize=None, frequency=[1e8], channel_bandwidth=numpy.array([1e6]),
                      phasecentre=None, polarisation_frame=Polarisation_Frame("stokesI")):
    """Create a useful test image

    This is the test image M31 widely used in ALMA and other simulations. It is actually part of an Halpha region in
    M31.

    :param canonical: Make the image into a 4 dimensional image
    :param cellsize:
    :param frequency: Frequency (array) in Hz
    :param channel_bandwidth: Channel bandwidth (array) in Hz
    :param phasecentre: Phase centre of image (SkyCoord)
    :param polarisation_frame: Polarisation frame
    :returns: Image
    """
    im = import_image_from_fits(arl_path("data/models/M31.MOD"))
    if canonical:
        if frequency is None:
            nchan = 1
        else:
            nchan = len(frequency)

        if polarisation_frame is None:
            im.polarisation_frame=Polarisation_Frame("stokesI")
        elif type(polarisation_frame) == Polarisation_Frame:
            im.polarisation_frame = polarisation_frame
        else:
            raise RuntimeError("polarisation_frame is not valid")
        
        im = replicate_image(im, frequency=frequency, polarisation_frame=im.polarisation_frame)
        if cellsize is not None:
            im.wcs.wcs.cdelt[0] = -180.0 * cellsize / numpy.pi
            im.wcs.wcs.cdelt[1] = +180.0 * cellsize / numpy.pi
        if frequency is not None:
            im.wcs.wcs.crval[3] = frequency[0]
        if channel_bandwidth is not None:
            im.wcs.wcs.cdelt[3] = channel_bandwidth[0]
        else:
            if len(frequency) > 1:
                im.wcs.wcs.cdelt[3] = frequency[1] - frequency[0]
            else:
                im.wcs.wcs.cdelt[3] = 0.001 * frequency[0]
        im.wcs.wcs.radesys = 'ICRS'
        im.wcs.wcs.equinox = 2000.00
        
    if phasecentre is not None:
        im.wcs.wcs.crval[0] = phasecentre.ra.deg
        im.wcs.wcs.crval[1] = phasecentre.dec.deg
        im.wcs.wcs.crpix[0] = im.data.shape[3] // 2
        im.wcs.wcs.crpix[1] = im.data.shape[2] // 2
    
    return im


def create_low_test_image(npixel=16384, polarisation_frame=Polarisation_Frame("stokesI"), cellsize=0.000015,
                          frequency=numpy.array([1e8]), channel_bandwidth=numpy.array([1e6]), phasecentre=None, fov=20):
    
    """Create LOW test image from S3
    
    The input catalog was generated at http://s-cubed.physics.ox.ac.uk/s3_sex using the following query::
        Database: s3_sex
        SQL: select * from Galaxies where (pow(10,itot_151)*1000 > 1.0) and (right_ascension between -5 and 5) and (declination between -5 and 5);;
    
    Number of rows returned: 29966
    
    There are three possible tables to use::
    
        data/models/S3_151MHz_10deg.csv, use fov=10
        data/models/S3_151MHz_20deg.csv, use fov=20
        data/models/S3_151MHz_40deg.csv, use fov=40
            
    The component spectral index is calculated from the 610MHz and 151MHz, and then calculated for the specified
    frequencies.
    
    If polarisation_frame is not stokesI then the image will a polarised axis but the values will be zero.
    
    :param npixel: Number of pixels
    :param polarisation_frame: Polarisation frame (default Polarisation_Frame("stokesI"))
    :param cellsize: cellsize in radians
    :param frequency:
    :param channel_bandwidth: Channel width (Hz)
    :param phasecentre: phasecentre (SkyCoord)
    :param fov: fov table to use
    :returns: Image
    """
    
    ras = []
    decs = []
    fluxes = []
    
    if phasecentre is None:
        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox=2000.0)

    if polarisation_frame is None:
        polarisation_frame = Polarisation_Frame("I")

    npol = polarisation_frame.npol
    
    nchan = len(frequency)
    
    shape = [nchan, npol, npixel, npixel]
    w = WCS(naxis=4)
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, 1.0, channel_bandwidth[0]]
    w.wcs.crpix = [npixel // 2, npixel // 2, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 1.0, frequency[0]]
    w.naxis = 4
    
    w.wcs.radesys = 'ICRS'
    w.wcs.equinox = 2000.0
    
    model = create_image_from_array(numpy.zeros(shape), w, polarisation_frame=polarisation_frame)
    
    
    assert fov in [10, 20, 40], "Field of view invalid: use one of %s" % ([10, 20, 40])
    with open(arl_path('data/models/S3_151MHz_%ddeg.csv' % (fov))) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        r = 0
        for row in readCSV:
            # Skip first row
            if r > 0:
                ra = float(row[4]) + phasecentre.ra.deg
                dec = float(row[5]) + phasecentre.dec.deg
                alpha = (float(row[10]) - float(row[9])) / numpy.log10(610.0 / 151.0)
                flux = numpy.power(10, float(row[9])) * numpy.power(frequency / 1.51e8, alpha)
                ras.append(ra)
                decs.append(dec)
                fluxes.append(flux)
            r += 1
    
    p = w.sub(2).wcs_world2pix(numpy.array(ras), numpy.array(decs), 1)
    total_flux = numpy.sum(fluxes)
    fluxes = numpy.array(fluxes)
    ip = numpy.round(p).astype('int')
    ok = numpy.where((0 <= ip[0, :]) & (npixel > ip[0, :]) & (0 <= ip[1, :]) & (npixel > ip[1, :]))[0]
    ps = ip[:, ok]
    fluxes = fluxes[ok]
    actual_flux = numpy.sum(fluxes)
    
    log.info('create_low_test_image: %d sources inside the image' % (ps.shape[1]))

    log.info('create_low_test_image: flux in S3 model = %.3f, actual flux in image = %.3f' % (total_flux, actual_flux))
    for chan in range(nchan):
        for iflux, flux in enumerate(fluxes):
            model.data[chan, 0, ps[1, iflux], ps[0, iflux]] = flux[chan]
    
    return model


def create_low_test_beam(model):
    """Create a test power beam for LOW using an image from OSKAR
    
    This is in progress. Currently uses the wrong beam!
    
    :param model: Template image
    :returns: Image
    """
    
    beam = import_image_from_fits(arl_path('data/models/SKA1_LOW_beam.fits'))
    
    # Scale the image cellsize to account for the different in frequencies. Eventually we will want to
    # use a frequency cube
    log.info("create_low_test_beam: primary beam is defined at %.3f MHz" % (beam.wcs.wcs.crval[2] * 1e-6))
    log.info("create_low_test_beam: scaling to model frequency %.3f MHz" % (model.wcs.wcs.crval[3] * 1e-6))
    
    
    nchan, npol, ny, nx = model.shape
    
    # We need to interpolate each frequency channel separately. The beam is assumed to just scale with
    # frequency.
    
    reprojected_beam = create_empty_image_like(model)

    for chan in range(nchan):
    
        model2dwcs = model.wcs.sub(2).deepcopy()
        model2dshape = [model.shape[2], model.shape[3]]
        beam2dwcs = beam.wcs.sub(2).deepcopy()
    
        # The frequency axis is the second to last in the beam
        frequency = model.wcs.sub(['spectral']).wcs_pix2world([chan], 0)[0]
        fscale = beam.wcs.wcs.crval[2] / frequency
        
        beam2dwcs.wcs.cdelt = fscale * beam.wcs.sub(2).wcs.cdelt
        beam2dwcs.wcs.crpix = beam.wcs.sub(2).wcs.crpix
        beam2dwcs.wcs.crval = model.wcs.sub(2).wcs.crval
        beam2dwcs.wcs.ctype = model.wcs.sub(2).wcs.ctype
        model2dwcs.wcs.crpix = [model.shape[2] // 2, model.shape[3] // 2]

        beam2d = create_image_from_array(beam.data[0,0,:,:], beam2dwcs)
        print(beam2dwcs)
        print(model2dwcs)
        reprojected_beam2d, footprint = reproject_image(beam2d, model2dwcs, shape=model2dshape)
        assert numpy.max(footprint.data) > 0.0, "No overlap between beam and model"
        
        reprojected_beam2d.data *= reprojected_beam2d.data
        reprojected_beam2d.data[footprint.data <= 0.0] = 0.0
        for pol in range(npol):
            reprojected_beam.data[chan, pol, :, :] = reprojected_beam2d.data[:,:]

    return reprojected_beam


def replicate_image(im: Image, polarisation_frame=Polarisation_Frame('stokesI'), frequency=1e8):
    """ Make a new canonical shape Image, extended along third and fourth axes by replication.

    The order of the data is [chan, pol, dec, ra]


    :param frequency:
    :param im:
    :param polarisation_frame: Polarisation_frame
    :param nchan: Number of spectral channels
    :returns: Image
    """
    
    if len(im.data.shape) == 2:
        fim = Image()
        
        newwcs = WCS(naxis=4)
        
        newwcs.wcs.crpix = [im.wcs.wcs.crpix[0], im.wcs.wcs.crpix[1], 1.0, 1.0]
        newwcs.wcs.cdelt = [im.wcs.wcs.cdelt[0], im.wcs.wcs.cdelt[1], 1.0, 1.0]
        newwcs.wcs.crval = [im.wcs.wcs.crval[0], im.wcs.wcs.crval[1], 1.0, frequency[0]]
        newwcs.wcs.ctype = [im.wcs.wcs.ctype[0], im.wcs.wcs.ctype[1], 'STOKES', 'FREQ']
        
        nchan = len(frequency)
        npol = polarisation_frame.npol
        fim.polarisation_frame = polarisation_frame
        
        fim.wcs = newwcs
        fshape = [nchan, npol, im.data.shape[1], im.data.shape[0]]
        fim.data = numpy.zeros(fshape)
        log.info("replicate_image: replicating shape %s to %s" % (im.data.shape, fim.data.shape))
        for i3 in range(nchan):
            fim.data[i3, 0, :, :] = im.data[:, :]
        return fim
    else:
        return im


def create_blockvisibility_iterator(config: Configuration, times: numpy.array, freq: numpy.array, phasecentre: SkyCoord,
                                    weight: float = 1, polarisation_frame=Polarisation_Frame('stokesI'),
                                    integration_time=1.0, number_integrations=1, channel_bandwidth=1e6,
                                    predict=predict_2d, model=None, components=None, phase_error=0.0,
                                    amplitude_error=0.0):
    """ Create a sequence of Visibiliites and optionally predicting and coalescing

    This is useful mainly for performing large simulations. Do something like::
    
        vis_iter = create_blockvisibility_iterator(config, times, frequency, phasecentre=phasecentre,
                                              weight=1.0, integration_time=30.0, number_integrations=3)

        for i, vis in enumerate(vis_iter):
        if i == 0:
            fullvis = vis
        else:
            fullvis = append_visibility(fullvis, vis)


    :param config: Configuration of antennas
    :param times: hour angles in radians
    :param freq: frequencies (Hz] Shape [nchan]
    :param weight: weight of a single sample
    :param phasecentre: phasecentre of observation
    :param npol: Number of polarizations
    :param integration_time: Integration time ('auto' or value in s)
    :param number_integrations: Number of integrations to be created at each time.
    :param model: Model image to be inserted
    :param components: Components to be inserted
    :returns: Visibility

    """
    for time in times:
        actualtimes = time + numpy.arange(0, number_integrations) * integration_time * numpy.pi / 43200.0
        vis = create_blockvisibility(config, actualtimes, freq=freq, phasecentre=phasecentre,
                                     polarisation_frame=polarisation_frame, weight=weight, integration_time=integration_time,
                                     channel_bandwidth=channel_bandwidth)
        
        if components is not None:
            vis = predict_skycomponent_blockvisibility(vis, components)
        
        # Add phase errors
        if phase_error > 0.0 or amplitude_error > 0.0:
            gt = create_gaintable_from_blockvisibility(vis)
            gt = simulate_gaintable(gt=gt, vis=vis, phase_error=phase_error, amplitude_error=amplitude_error)
            original = copy_visibility(vis)
            vis = apply_gaintable(vis, gt)
        
        yield vis


def simulate_gaintable(gt: GainTable, phase_error=0.1, amplitude_error=0.0, **kwargs):
    """ Simulate a gain table
    
    :type gt: GainTable
    :param phase_error: std of normal distribution, zero mean
    :param amplitude_error: std of log normal distribution
    
    """
    log.info("simulate_gaintable: Simulating amplitude error = %.4f, phase error = %.4f"
             % (amplitude_error, phase_error))
    amp = 1.0
    phasor = 1.0
    if phase_error:
        phasor = numpy.exp(1j * numpy.random.normal(0, phase_error, gt.data['gain'].shape))
    if amplitude_error > 0.0:
        amp = numpy.random.lognormal(mean=0.0, sigma=amplitude_error, size=gt.data['gain'].shape)
        
    gt.data['gain'] = amp * phasor

    return gt
