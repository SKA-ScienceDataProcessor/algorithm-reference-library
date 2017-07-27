""" Visibility operations

"""

import logging
from typing import Union

import numpy
from astropy.coordinates import SkyCoord

from arl.data.data_models import BlockVisibility, Visibility, QA
from arl.imaging.params import get_frequency_map
from arl.util.coordinate_support import xyz_to_uvw, uvw_to_xyz, skycoord_to_lmn, simulate_point
from arl.visibility.base import copy_visibility

log = logging.getLogger(__name__)


def append_visibility(vis: Union[Visibility, BlockVisibility], othervis: Union[Visibility, BlockVisibility]) \
        -> Union[Visibility, BlockVisibility]:
    """Append othervis to vis
    
    :param vis:
    :param othervis:
    :return: Visibility vis + othervis
    """
    assert vis.polarisation_frame == othervis.polarisation_frame
    assert vis.phasecentre == othervis.phasecentre
    vis.data = numpy.hstack((vis.data, othervis.data))
    return vis


def sum_visibility(vis: Visibility, direction: SkyCoord) -> numpy.array:
    """ Direct Fourier summation in a given direction

    :param vis: Visibility to be summed
    :param direction: Direction of summation
    :return: flux[nch,npol], weight[nch,pol]
    """
    # TODO: Convert to Visibility or remove?
    
    svis = copy_visibility(vis)
    
    l, m, n = skycoord_to_lmn(direction, svis.phasecentre)
    phasor = numpy.conjugate(simulate_point(svis.uvw, l, m))
    
    # Need to put correct mapping here
    _, frequency = get_frequency_map(svis, None)
    
    frequency = list(frequency)
    
    nchan = max(frequency) + 1
    npol = svis.polarisation_frame.npol
    
    flux = numpy.zeros([nchan, npol])
    weight = numpy.zeros([nchan, npol])
    
    coords = svis.vis, svis.weight, phasor, list(frequency)
    for v, wt, p, ic in zip(*coords):
        for pol in range(npol):
            flux[ic, pol] += numpy.real(wt[pol] * v[pol] * p)
            weight[ic, pol] += wt[pol]
    
    flux[weight > 0.0] = flux[weight > 0.0] / weight[weight > 0.0]
    flux[weight <= 0.0] = 0.0
    return flux, weight


def qa_visibility(vis: Union[Visibility, BlockVisibility], context=None) -> QA:
    """Assess the quality of Visibility

    :param context:
    :param vis: Visibility to be assessed
    :return: QA
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


def remove_continuum_blockvisibility(vis: BlockVisibility, degree=1, mask=None, **kwargs) -> BlockVisibility:
    """ Fit and remove continuum visibility

    Fit a polynomial in frequency of the specified degree where mask is True
  
    :param vis:
    :param degree: Degree of polynomial
    :param mask:
    :param kwargs:
    :return:
    """
    if mask is not None:
        assert numpy.sum(mask) > 2 * degree, "Insufficient channels for fit"
    
    nchan = len(vis.frequency)
    x = (vis.frequency - vis.frequency[nchan // 2]) / (vis.frequency[0] - vis.frequency[nchan // 2])
    for row in range(vis.nvis):
        for ant2 in range(vis.nants):
            for ant1 in range(vis.nants):
                for pol in range(vis.polarisation_frame.npol):
                    wt = numpy.sqrt(vis.data['weight'][row, ant2, ant1, :, pol])
                    if mask is not None:
                        wt[mask] = 0.0
                    fit = numpy.polyfit(x, vis.data['vis'][row, ant2, ant1, :, pol], w=wt, deg=degree)
                    prediction = numpy.polyval(fit, x)
                    vis.data['vis'][row, ant2, ant1, :, pol] -= prediction
    return vis


def divide_visibility(vis: BlockVisibility, modelvis: BlockVisibility):
    """ Divide visibility by model forming visibility for equivalent point source

    This is a useful intermediate product for calibration. Variation of the visibility in time and
    frequency due to the model structure is removed and the data can be averaged to a limit determined
    by the instrumental stability. The weight is adjusted to compensate for the division.
    
    Zero divisions are avoided and the corresponding weight set to zero.

    :param vis:
    :param modelvis:
    :return:
    """
    # Different for scalar and vector/matrix cases
    isscalar = vis.polarisation_frame.npol == 1
    
    if isscalar:
        # Scalar case is straightforward
        x = numpy.zeros_like(vis.vis)
        xwt = numpy.abs(modelvis.vis) ** 2 * vis.weight
        mask = xwt > 0.0
        x[mask] = vis.vis[mask] / modelvis.vis[mask]
    else:
        nrows, nants, _, nchan, npol = vis.vis.shape
        nrec = 2
        assert nrec * nrec == npol
        xshape = (nrows, nants, nants, nchan, nrec, nrec)
        x = numpy.zeros(xshape, dtype='complex')
        xwt = numpy.zeros(xshape)
        for row in range(nrows):
            for ant1 in range(nants):
                for ant2 in range(ant1 + 1, nants):
                    for chan in range(nchan):
                        ovis = numpy.matrix(vis.vis[row, ant2, ant1, chan].reshape([2, 2]))
                        mvis = numpy.matrix(modelvis.vis[row, ant2, ant1, chan].reshape([2, 2]))
                        wt = numpy.matrix(vis.weight[row, ant2, ant1, chan].reshape([2, 2]))
                        x[row, ant2, ant1, chan] = numpy.matmul(numpy.linalg.inv(mvis), ovis)
                        xwt[row, ant2, ant1, chan] = numpy.dot(mvis, numpy.multiply(wt, mvis.H)).real
        x = x.reshape((nrows, nants, nants, nchan, nrec * nrec))
        xwt = xwt.reshape((nrows, nants, nants, nchan, nrec * nrec))
    
    pointsource_vis = BlockVisibility(data=None, frequency=vis.frequency, channel_bandwidth=vis.channel_bandwidth,
                                      phasecentre=vis.phasecentre, configuration=vis.configuration,
                                      uvw=vis.uvw, time=vis.time, integration_time=vis.integration_time, vis=x,
                                      weight=xwt)
    return pointsource_vis


def integrate_visibility_by_channel(vis: BlockVisibility, **kwargs) -> BlockVisibility:
    """ Integrate visibility across channels, returning new visibility
    
    :param vis:
    :param kwargs:
    :return: BlockVisibility
    """
    
    vis_shape = list(vis.vis.shape)
    ntimes, nants, _, nchan, npol = vis_shape
    vis_shape[-2] = 1
    newvis = BlockVisibility(data=None,
                             frequency=numpy.ones([1]) * numpy.average(vis.frequency),
                             channel_bandwidth=numpy.ones([1]) * numpy.sum(vis.channel_bandwidth),
                             phasecentre=vis.phasecentre,
                             configuration=vis.configuration,
                             uvw=vis.uvw,
                             time=vis.time,
                             vis=numpy.zeros(vis_shape, dtype='complex'),
                             weight=numpy.ones(vis_shape, dtype='float'),
                             integration_time=vis.integration_time,
                             polarisation_frame=vis.polarisation_frame)
    
    newvis.data['vis'][..., 0, :] = numpy.sum(vis.data['vis'] * vis.data['weight'], axis=-2)
    newvis.data['weight'][..., 0, :] = numpy.sum(vis.data['weight'], axis=-2)
    mask = newvis.data['weight'] > 0.0
    newvis.data['vis'][mask] = newvis.data['vis'][mask] / newvis.data['weight'][mask]
    
    return newvis
