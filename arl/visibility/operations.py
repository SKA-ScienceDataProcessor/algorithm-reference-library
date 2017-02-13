# Tim Cornwell <realtimcornwell@gmail.com>
#
""" BlockVisibility operations

"""

import copy

import astropy.constants as constants

from arl.data.parameters import *
from arl.fourier_transforms.ftprocessor_params import *
from arl.util.coordinate_support import *

log = logging.getLogger(__name__)


def vis_summary(vis):
    """Return string summarizing the BlockVisibility
    
    """
    return "Visibility: %d rows, %.3f GB" % (vis.nvis, vis.size())


def copy_visibility(vis):
    """Copy a visibility
    
    Performs a deepcopy of the data array
    """
    newvis = copy.copy(vis)
    newvis.data = copy.deepcopy(vis.data)
    return newvis


def create_blockvisibility(config: Configuration, times: numpy.array, freq: numpy.array, phasecentre: SkyCoord,
                      weight: float, meta: dict = None, npol=4, integration_time=1.0) -> BlockVisibility:
    """ Create a BlockVisibility from Configuration, hour angles, and direction of source
    
    Note that we keep track of the integration time for BDA purposes

    :param params:
    :param config: Configuration of antennas
    :param times: hour angles in radians
    :param freq: frequencies (Hz] Shape [nchan]
    :param weight: weight of a single sample
    :param phasecentre: phasecentre of observation
    :param npol: Number of polarizations
    :param integration_time: Integration time ('auto' or value in s)
    :returns: BlockVisibility
    """
    assert phasecentre is not None, "Must specify phase centre"
    nch = len(freq)
    ants_xyz = config.data['xyz']
    nants = len(config.data['names'])
    nbaselines = int(nants * (nants - 1) / 2)
    ntimes = len(times)
    nrows = nbaselines * ntimes
    row = 0
    rvis = numpy.zeros([nrows, nch, npol], dtype='complex')
    rweight = weight * numpy.ones([nrows, nch, npol])
    rtimes = numpy.zeros([nrows])
    rantenna1 = numpy.zeros([nrows], dtype='int')
    rantenna2 = numpy.zeros([nrows], dtype='int')
    for ha in times:
        rtimes[row:row + nbaselines] = ha * 43200.0 / numpy.pi
        for a1 in range(nants):
            for a2 in range(a1 + 1, nants):
                rantenna1[row] = a1
                rantenna2[row] = a2
                row += 1
    ruvw = xyz_to_baselines(ants_xyz, times, phasecentre.dec.rad)
    rintegration_time = numpy.full_like(rtimes, integration_time)
    vis = BlockVisibility(uvw=ruvw, time=rtimes, antenna1=rantenna1, antenna2=rantenna2, vis=rvis, weight=rweight,
                          imaging_weight=rweight, integration_time=rintegration_time)
    vis.frequency = freq
    vis.phasecentre = phasecentre
    vis.configuration = config
    log.info("create_blockvisibility: %s" % (vis_summary(vis)))
    return vis


def create_visibility(config: Configuration, times: numpy.array, freq: numpy.array,
                                phasecentre: SkyCoord, weight: float, npol=4,
                                pol_frame=Polarisation_Frame.linear,
                                integration_time=1.0) -> Visibility:
    """ Create a BlockVisibility from Configuration, hour angles, and direction of source

    Note that we keep track of the integration time for BDA purposes

    :param params:
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
    for iha, ha in enumerate(times):
        ant_pos = xyz_to_uvw(ants_xyz, ha, phasecentre.dec.value)
        rtimes[row:row + nbaselines * npol * nch] = ha * 43200.0 / numpy.pi
        for a1 in range(nants):
            for a2 in range(a1 + 1, nants):
                rantenna1[row:row + npol * nch] = a1
                rantenna2[row:row + npol * nch] = a2
                for ch in range(nch):
                    k = freq[ch] / constants.c.value
                    ruvw[row:row + npol, :] = (ant_pos[a1,:] - ant_pos[a2,:]) * k
                    rpolarisation[row:row + npol] = range(npol)
                    rfrequency[row:row + npol] = freq[ch]
                    row += npol
    
    assert row == nrows
    rintegration_time = numpy.full_like(rtimes, integration_time)
    vis = Visibility(uvw=ruvw, time=rtimes, antenna1=rantenna1, antenna2=rantenna2,
                               frequency=rfrequency, polarisation=rpolarisation, vis=rvis,
                               weight=rweight, imaging_weight=rweight, integration_time=rintegration_time,
                               polarisation_frame=pol_frame)
    vis.phasecentre = phasecentre
    vis.configuration = config
    log.info("create_visibility: %s" % (vis_summary(vis)))
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis
    
    return vis


def create_blockvisibility_from_rows(vis: BlockVisibility, rows, makecopy=True) -> BlockVisibility:
    """ Create a BlockVisibility from selected rows

    :param params:
    :param vis: BlockVisibility
    :param rows: Boolean array of row selction
    :param makecopy: Make a deep copy (True)
    :returns: BlockVisibility
    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis
    
    if makecopy:
        newvis = copy_visibility(vis)
        newvis.data = copy.deepcopy(vis.data[rows])
        log.info("create_visibility_from_rows: Created new compressed visibility table")
        assert len(newvis.data) == numpy.sum(rows)
        return newvis
    else:
        vis.data = copy.deepcopy(vis.data[rows])
        log.info("create_visibility_from_rows: Created view into compressed visibility table")
        assert len(vis.data) == numpy.sum(rows)
        return vis


def create_visibility_from_rows(vis: Visibility, rows, makecopy=True) -> Visibility:
    """ Create a BlockVisibility from selected rows

    :param params:
    :param vis: Visibility
    :param rows: Boolean array of row selction
    :param makecopy: Make a deep copy (True)
    :returns: Visibility
    """
    
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis
    
    if makecopy:
        newvis = copy_visibility(vis)
        newvis.data = copy.deepcopy(vis.data[rows])
        log.info("create_visibility_from_rows: Created new compressed visibility table")
        assert len(newvis.data) == numpy.sum(rows)
        return newvis
    else:
        vis.data = copy.deepcopy(vis.data[rows])
        log.info("create_visibility_from_rows: Created view into compressed visibility table")
        assert len(vis.data) == numpy.sum(rows)
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
        newvis = copy.copy(vis)
        
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


def sum_visibility(vis: BlockVisibility, direction: SkyCoord) -> numpy.array:
    """ Direct Fourier summation in a given direction

    :param params:
    :param vis: BlockVisibility to be summed
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
    """Assess the quality of BlockVisibility

    :param params:
    :param vis: BlockVisibility to be assessed
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
