# Tim Cornwell <realtimcornwell@gmail.com>
#
""" Visibility operations

"""

import copy

from arl.calibration.stefcal import stefcal
from arl.fourier_transforms.ftprocessor_params import *
from arl.util.coordinate_support import *
from arl.visibility.iterators import vis_timeslice_iter
from arl.visibility.operations import create_blockvisibility_from_rows

log = logging.getLogger(__name__)


def gaintable_summary(gt):
    """Return string summarizing the Gaintable

    """
    return "%s rows, %.3f GB" % (gt.data.shape, gt.size())


def create_gaintable_from_blockvisibility(vis: BlockVisibility, time_width: float = None, frequency_width: float = None,
                                          **kwargs):
    """ Create gain table from visibility
    
    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis
    
    nants = vis.nants
    utimes = numpy.unique(vis.time)
    ntimes = len(utimes)
    ufrequency = numpy.unique(vis.frequency)
    nfrequency = len(ufrequency)
    
    npolarisation = vis.polarisation_frame.npol
    
    gainshape = [ntimes, nants, nfrequency, npolarisation]
    gain = numpy.ones(gainshape, dtype='complex')
    gain_weight = numpy.ones(gainshape)
    gain_time = utimes
    gain_frequency = ufrequency
    gain_antenna = range(nants)
    
    gt = GainTable(gain=gain, time=gain_time, antenna=gain_antenna, weight=gain_weight, frequency=gain_frequency,
                   polarisation_frame=vis.polarisation_frame)
    
    return gt


def apply_gaintable(vis: BlockVisibility, gt: GainTable, inverse=False):
    """Apply a gain table to a block visibility
    
    :param vis: Visibility to have gains applied
    :param gt: Gaintable to be applied
    :param inverse: Apply the inverse (default=False)
    :returns: input vis with gains applied
    
    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis
    
    if inverse:
        log.info('apply_gaintable: Apply inverse gaintable')
    else:
        log.info('apply_gaintable: Apply gaintable')
    
    if vis.polarisation_frame.type == Polarisation_Frame('stokesI').type:
        for chunk, rows in enumerate(vis_timeslice_iter(vis)):
            visslice = create_blockvisibility_from_rows(vis, rows)
            vistime = numpy.average(visslice.time)
            integration_time = numpy.average(visslice.integration_time)
            gaintable_rows = abs(gt.time - vistime) < integration_time / 2.0
            
            # Lookup the gain for this set of visibilities
            gain = gt.data['gain'][gaintable_rows]
            gwt = gt.data['weight'][gaintable_rows]
            if inverse:  # TODO: Make this true inverse for polarisation
                gain[gwt > 0.0] = 1.0 / gain[gwt > 0.0]
            
            original = visslice.data['vis']
            applied = copy.deepcopy(original)
            for a1 in range(visslice.nants - 1):
                for a2 in range(a1 + 1, visslice.nants):
                    applied[:, a2, a1, :, :] = gain[:, a1, :, :] * numpy.conjugate(gain[:, a2, :, :]) * \
                                               original[:, a2, a1, :, :]
            
            vis.data['vis'][rows] = applied
    return vis


def solve_gaintable(vis: BlockVisibility, modelvis: BlockVisibility, phase_only=True):
    """Solve a gain table to a block visibility

    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis
    
    if phase_only:
        log.info('solve_gaintable: Solving for phase only')
    else:
        log.info('solve_gaintable: Solving for complex gain')
    
    gt = create_gaintable_from_blockvisibility(vis)
    
    if vis.polarisation_frame.type == Polarisation_Frame('stokesI').type:
        for chunk, rows in enumerate(vis_timeslice_iter(vis)):
            visslice = create_blockvisibility_from_rows(vis, rows)
            mvisslice = create_blockvisibility_from_rows(modelvis, rows)
            
            # Form the point source equivalent visibility
            X = numpy.zeros_like(visslice.data['vis'])
            Xwt = numpy.abs(mvisslice.data['vis']) ** 2 * mvisslice.data['weight']
            mask = Xwt > 0.0
            X[mask] = visslice.data['vis'][mask] / mvisslice.data['vis'][mask]
            
            # Now average over time, chan. The axes of X are time, antenna2, antenna1, chan, pol
            
            Xave = numpy.average(X * Xwt, axis=(0))
            XwtAve = numpy.average(Xwt, axis=(0))
            
            mask = XwtAve > 0.0
            Xave[mask] = Xave[mask] / XwtAve[mask]
            
            gt.data['gain'][chunk, ...], gt.data['weight'][chunk, ...], residual = \
                solve_station_gains_itsubs(Xave, XwtAve, phase_only=phase_only)
    
    return gt


def solve_station_gains_itsubs(X, Xwt, niter=30, tol=1e-12, phase_only=True, refant=0):
    """Solve for the antenna gains
    
    X(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))
    
    This uses an iterative substitution algorithm due to Larry D'Addario c 1980'ish. Used
    in the original Dec-10 Antsol
    
    :param X: Equivalent point source visibility[ nants, nants, ...]
    :param Xwt: Equivalent point source weight [nants, nants, ...]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :returns: gain [nants, ...], weight [nants, ...]
    """
    
    nants = X.shape[0]
    for ant1 in range(nants):
        X[ant1, ant1, ...] = 0.0
        Xwt[ant1, ant1, ...] = 0.0
        for ant2 in range(ant1 + 1, nants):
            X[ant1, ant2, ...] = numpy.conjugate(X[ant2, ant1, ...])
            Xwt[ant1, ant2, ...] = Xwt[ant2, ant1, ...]
    
    def gain_substitution(gain, X, Xwt):
        
        nants = gain.shape[0]
        g = numpy.ones_like(gain, dtype='complex')
        gwt = numpy.zeros_like(gain, dtype='float')
        
        for ant1 in range(nants):
            top = numpy.sum(gain[:, ...] * X[:, ant1, ...] * Xwt[:, ant1, ...], axis=0)
            bot = numpy.sum((gain[:, ...] * numpy.conjugate(gain[:, ...])).real * Xwt[:, ant1, ...], axis=0)
            g[ant1, ...] = top / bot
            gwt[ant1, ...] = bot
        return g, gwt
    
    
    gainshape = X.shape[1:]
    gain = numpy.ones(shape=gainshape, dtype=X.dtype)
    for iter in range(niter):
        gainLast = gain
        gain, gwt = gain_substitution(gain, X, Xwt)
        if phase_only:
            gain = gain / numpy.abs(gain)
        gain *= numpy.conjugate(gain[refant, ...]) / numpy.abs(gain[refant, ...])
        gain = 0.5 * (gain + gainLast)
        change = numpy.max(numpy.abs(gain - gainLast))
        if change < tol:
            residual = solution_residual(gain, X, Xwt)
            return gain, gwt, residual
        residual = solution_residual(gain, X, Xwt)
    
    return gain, gwt, residual

def solution_residual(gain, X, Xwt):
    """Calculate residual across all baselines of gain for point source equivalent visibilities
    
    :param gain: gain [nant, ...]
    :param X: Point source equivalent visibility [nant, ...]
    :param Xwt: Point source equivalent weight [nant, ...]
    :returns: residual[...]
    """

    nants = gain.shape[0]

    residual = 0.0
    sumwt = 0.0
    for ant1 in range(nants):
        # residual += numpy.abs(X[ant2, ant1, ...] - gain[ant1, ...] * numpy.conjugate(gain[ant2, ...])) ** 2 \
        #             * Xwt[ant2, ant1, ...]
        sumwt += numpy.sum(Xwt[:, ant1, ...], axis=0)
        residual += numpy.sum(numpy.abs(X[:, ant1, ...] - gain[ant1, ...] * numpy.conjugate(gain[:, ...])) ** 2 \
                              * Xwt[:, ant1, ...], axis=0)
    residual = numpy.sqrt(residual / sumwt)
    return residual


def solve_station_gains_stefcal(X, Xwt, niter=10, tol=1e-12, phase_only=True, refant=0):
    """Solve for the antenna gains

    X(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))

    This uses Stefcal

    :param X: Equivalent point source visibility[ nants, nants, ...]
    :param Xwt: Equivalent point source weight [nants, nants, ...]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :returns: gain [nants, ...], weight [nants, ...]
    """
    
    nants = X.shape[0]
    for ant1 in range(nants):
        X[ant1, ant1, ...] = 0.0
        Xwt[ant1, ant1, ...] = 0.0
        for ant2 in range(ant1 + 1, nants):
            X[ant1, ant2, ...] = numpy.conjugate(X[ant2, ant1, ...])
            Xwt[ant1, ant2, ...] = Xwt[ant2, ant1, ...]
        
    _, nants, nchan, npol = X.shape
    antA = enumerate(range(nants))
    antB = enumerate(range(nants))
    gain = numpy.ones(shape=[nants, nchan, npol], dtype='complex')
    gwt = numpy.zeros(shape=[nants, nchan, npol])
    for chan in range(nchan):
        for pol in range(npol):
            gain[:, chan, pol] = stefcal(X, nants, antA, antB, weights=1.0, num_iters=niter, ref_ant=refant,
                                         init_gain=None)
    residual = solution_residual(gain, X, Xwt)
    
    return gain, gwt, residual
