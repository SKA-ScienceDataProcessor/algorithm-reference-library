# Tim Cornwell <realtimcornwell@gmail.com>
#
""" Visibility operations

"""

import copy

import astropy.constants as constants

from arl.fourier_transforms.ftprocessor_params import *
from arl.util.coordinate_support import *

log = logging.getLogger(__name__)


def vis_summary(vis):
    """Return string summarizing the Visibility
    
    """
    return "%d rows, %.3f GB" % (vis.nvis, vis.size())

def append_visibility(vis: Visibility, othervis: Visibility):
    """Append othervis to vis
    
    :param vis:
    :param othervis:
    :returns: Visibility vis + othervis
    """
    assert vis.polarisation_frame == othervis.polarisation_frame
    assert vis.phasecentre == othervis.phasecentre
    vis.data = numpy.hstack((vis.data, othervis.data))
    return vis


def copy_visibility(vis):
    """Copy a visibility
    
    Performs a deepcopy of the data array
    """
    newvis = copy.copy(vis)
    newvis.data = copy.deepcopy(vis.data)
    return newvis


def create_visibility(config: Configuration, times: numpy.array, freq: numpy.array,
                      phasecentre: SkyCoord, weight: float, npol=4,
                      pol_frame=Polarisation_Frame('stokesI'),
                      integration_time=1.0, channel_bandwidth=1e6) -> Visibility:
    """ Create a Visibility from Configuration, hour angles, and direction of source

    Note that we keep track of the integration time for BDA purposes

    :param config: Configuration of antennas
    :param times: hour angles in radians
    :param freq: frequencies (Hz] Shape [nchan]
    :param weight: weight of a single sample
    :param phasecentre: phasecentre of observation
    :param npol: Number of polarizations
    :param integration_time: Integration time ('auto' or value in s)
    :returns: Visibility
    """
    assert phasecentre is not None, "Must specify phase centre"
    nch = len(freq)
    ants_xyz = config.data['xyz']
    nants = len(config.data['names'])
    nbaselines = int(nants * (nants - 1) / 2)
    ntimes = len(times)
    nrows = nbaselines * ntimes * nch * npol
    nrowsperintegration = nbaselines * nch * npol
    row = 0
    rvis = numpy.zeros([nrows], dtype='complex')
    rweight = weight * numpy.ones([nrows])
    rtimes = numpy.zeros([nrows])
    rfrequency = numpy.zeros([nrows])
    rpolarisation = numpy.zeros([nrows], dtype='int')
    rantenna1 = numpy.zeros([nrows], dtype='int')
    rantenna2 = numpy.zeros([nrows], dtype='int')
    ruvw = numpy.zeros([nrows, 3])
    
    # Do each hour angle in turn
    for iha, ha in enumerate(times):
        
        # Calculate the positions of the antennas as seen for this hour angle
        # and declination
        ant_pos = xyz_to_uvw(ants_xyz, ha, phasecentre.dec.rad)
        rtimes[row:row + nrowsperintegration] = ha * 43200.0 / numpy.pi
        
        # Loop over all pairs of antennas. Note that a2>a1
        for a1 in range(nants):
            for a2 in range(a1 + 1, nants):
                rantenna1[row:row + npol * nch] = a1
                rantenna2[row:row + npol * nch] = a2
                
                # Loop over all frequencies and polarisations
                for ch in range(nch):
                    # noinspection PyUnresolvedReferences
                    k = freq[ch] / constants.c.value
                    ruvw[row:row + npol, :] = (ant_pos[a2, :] - ant_pos[a1, :]) * k
                    rpolarisation[row:row + npol] = range(npol)
                    rfrequency[row:row + npol] = freq[ch]
                    row += npol
    
    assert row == nrows
    rintegration_time = numpy.full_like(rtimes, integration_time)
    rchannel_bandwidth = numpy.full_like(rfrequency, channel_bandwidth)
    vis = Visibility(uvw=ruvw, time=rtimes, antenna1=rantenna1, antenna2=rantenna2,
                     frequency=rfrequency, polarisation=rpolarisation, vis=rvis,
                     weight=rweight, imaging_weight=rweight,
                     integration_time=rintegration_time, channel_bandwidth=rchannel_bandwidth,
                     polarisation_frame=pol_frame)
    vis.phasecentre = phasecentre
    vis.configuration = config
    log.info("create_visibility: %s" % (vis_summary(vis)))
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis
    
    return vis


def create_blockvisibility(config: Configuration, times: numpy.array, freq: numpy.array,
                      phasecentre: SkyCoord, weight: float, npol=4,
                      pol_frame=Polarisation_Frame('stokesI'),
                      integration_time=1.0, channel_bandwidth=1e6) -> Visibility:
    """ Create a BlockVisibility from Configuration, hour angles, and direction of source

    Note that we keep track of the integration time for BDA purposes

    :param config: Configuration of antennas
    :param times: hour angles in radians
    :param freq: frequencies (Hz] Shape [nchan]
    :param weight: weight of a single sample
    :param phasecentre: phasecentre of observation
    :param npol: Number of polarizations
    :param integration_time: Integration time ('auto' or value in s)
    :returns: Visibility
    """
    assert phasecentre is not None, "Must specify phase centre"
    nch = len(freq)
    ants_xyz = config.data['xyz']
    nants = len(config.data['names'])
    nbaselines = int(nants * (nants - 1) / 2)
    ntimes = len(times)
    visshape = [ntimes, nants, nants, nch, npol]
    rvis = numpy.zeros(visshape, dtype='complex')
    rweight = weight * numpy.ones(visshape)
    rtimes = numpy.zeros([ntimes])
    ruvw = numpy.zeros([ntimes, nants, nants, 3])
    
    # Do each hour angle in turn
    for iha, ha in enumerate(times):
        
        # Calculate the positions of the antennas as seen for this hour angle
        # and declination
        ant_pos = xyz_to_uvw(ants_xyz, ha, phasecentre.dec.rad)
        rtimes[iha] = ha * 43200.0 / numpy.pi
        
        # Loop over all pairs of antennas. Note that a2>a1
        for a1 in range(nants):
            for a2 in range(a1 + 1, nants):
                ruvw[iha, a2, a1, :] = (ant_pos[a2, :] - ant_pos[a1, :])
    
    rintegration_time = numpy.full_like(rtimes, integration_time)
    rchannel_bandwidth = numpy.full_like(freq, channel_bandwidth)
    vis = BlockVisibility(uvw=ruvw, time=rtimes, frequency=freq, vis=rvis, weight=rweight,
                          integration_time=rintegration_time, channel_bandwidth=rchannel_bandwidth,
                          polarisation_frame=pol_frame)
    vis.phasecentre = phasecentre
    vis.configuration = config
    log.info("create_blockvisibility: %s" % (vis_summary(vis)))
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis
    
    return vis


def create_visibility_from_rows(vis: Visibility, rows, makecopy=True) -> Visibility:
    """ Create a Visibility from selected rows

    :param vis: Visibility
    :param rows: Boolean array of row selction
    :param makecopy: Make a deep copy (True)
    :returns: Visibility
    """
    
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis
    
    if makecopy:
        newvis = copy_visibility(vis)
        newvis.data = copy.deepcopy(vis.data[rows])
        return newvis
    else:
        vis.data = copy.deepcopy(vis.data[rows])
        return vis


def create_blockvisibility_from_rows(vis: BlockVisibility, rows, makecopy=True) -> Visibility:
    """ Create a BlockVisibility from selected rows

    :param vis: BlockVisibility
    :param rows: Boolean array of row selction
    :param makecopy: Make a deep copy (True)
    :returns: Visibility
    """
    
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis
    
    if makecopy:
        newvis = copy_visibility(vis)
        newvis.data = copy.deepcopy(vis.data[rows])
        newvis.time = copy.deepcopy(vis.time[rows])

        return newvis
    else:
        vis.data = copy.deepcopy(vis.data[rows])
        vis.time = copy.deepcopy(vis.time[rows])
        
        return vis


def phaserotate_visibility(vis: Visibility, newphasecentre: SkyCoord, tangent=True,
                                     inverse=False) -> Visibility:
    """
    Phase rotate from the current phase centre to a new phase centre

    :param vis: Visibility to be rotated
    :param newphasecentre:
    :param tangent: Stay on the same tangent plane? (True)
    :param inverse: Actually do the opposite
    :returns: Visibility
    """
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis
    
    l, m, n = skycoord_to_lmn(newphasecentre, vis.phasecentre)
    
    # No significant change?
    if numpy.abs(l) > 1e-15 or numpy.abs(m) > 1e-15:
        
        # Make a new copy
        newvis = copy_visibility(vis)
        
        phasor = simulate_point(newvis.uvw, l, m)
        
        if inverse:
            newvis.data['vis'] *= phasor
        else:
            newvis.data['vis'] *= numpy.conj(phasor)
        
        # To rotate UVW, rotate into the global XYZ coordinate system and back. We have the option of
        # staying on the tangent plane or not. If we stay on the tangent then the raster will
        # join smoothly at the edges. If we change the tangent then we will have to reproject to get
        # the results on the same image, in which case overlaps or gaps are difficult to deal with.
        if not tangent:
            if inverse:
                xyz = uvw_to_xyz(vis.data['uvw'], ha=-newvis.phasecentre.ra.rad, dec=newvis.phasecentre.dec.rad)
                newvis.data['uvw'][...] = \
                    xyz_to_uvw(xyz, ha=-newphasecentre.ra.rad, dec=newphasecentre.dec.rad)[...]
            else:
                # This is the original (non-inverse) code
                xyz = uvw_to_xyz(newvis.data['uvw'], ha=-newvis.phasecentre.ra.rad, dec=newvis.phasecentre.dec.rad)
                newvis.data['uvw'][...] = xyz_to_uvw(xyz, ha=-newphasecentre.ra.rad, dec=newphasecentre.dec.rad)[
                    ...]
        newvis.phasecentre = newphasecentre
    else:
        newvis = vis
        log.warning("phaserotate_visibility: Null phase rotation")
    
    return newvis


def sum_visibility(vis: Visibility, direction: SkyCoord) -> numpy.array:
    """ Direct Fourier summation in a given direction

    :param vis: Visibility to be summed
    :param direction: Direction of summation
    :returns: flux[nch,npol], weight[nch,pol]
    """
    # TODO: Convert to Visibility or remove?
    
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis
    
    l, m, n = skycoord_to_lmn(direction, vis.phasecentre)
    phasor = numpy.conjugate(simulate_point(vis.uvw, l, m))
    
    # Need to put correct mapping here
    _, polarisations = get_polarisation_map(vis)
    _, frequency = get_frequency_map(vis)
    
    frequency = list(frequency)
    polarisations = list(polarisations)
    
    nchan = max(frequency) + 1
    npol = max(polarisations) + 1
    
    flux = numpy.zeros([nchan, npol])
    weight = numpy.zeros([nchan, npol])
    
    coords = vis.vis, vis.weight, phasor, list(frequency), list(polarisations)
    for v, wt, p, ic, ip in zip(*coords):
        flux[ic, ip] += numpy.real(wt * v * p)
        weight[ic, ip] += wt
    
    flux[weight > 0.0] = flux[weight > 0.0] / weight[weight > 0.0]
    flux[weight <= 0.0] = 0.0
    return flux, weight


def qa_visibility(vis, context=None):
    """Assess the quality of Visibility

    :param vis: Visibility to be assessed
    :returns: AQ
    """
    avis = numpy.abs(vis.vis)
    data = {'maxabs': numpy.max(avis),
            'minabs': numpy.min(avis),
            'rms': numpy.std(avis),
            'medianabs': numpy.median(avis)}
    qa = QA(origin=None,
            data=data,
            context=context)
    return qa
