# Tim Cornwell <realtimcornwell@gmail.com>
#
""" Visibility operations

"""

import copy

import numpy.linalg

from arl.fourier_transforms.ftprocessor_params import *
from arl.util.coordinate_support import *
from arl.visibility.iterators import vis_timeslice_iter

log = logging.getLogger(__name__)


def gaintable_summary(gt):
    """Return string summarizing the Gaintable

    """
    return "%s rows, %.3f GB" % (gt.data.shape, gt.size())


def create_gaintable_from_blockvisibility(vis: BlockVisibility, time_width: float = None,
                                          frequency_width: float = None) -> GainTable:
    """ Create gain table from visibility.
    
    This makes an empty gain table consistent with the BlockVisibility.
    
    :param vis: BlockVisibilty
    :param time_width: Time interval between solutions (s)
    :param frequency_width: Frequency solution width (Hz)
    :returns: GainTable
    
    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis
    
    nants = vis.nants
    utimes = numpy.unique(vis.time)
    ntimes = len(utimes)
    ufrequency = numpy.unique(vis.frequency)
    nfrequency = len(ufrequency)
    
    receptor_frame = ReceptorFrame(vis.polarisation_frame.type)
    nrec = receptor_frame.nrec
    vnpol = vis.polarisation_frame.npol
    
    gainshape = [ntimes, nants, nfrequency, nrec, nrec]
    gain = numpy.ones(gainshape, dtype='complex')
    if nrec > 1:
        gain[..., 0, 1] = 0.0
        gain[..., 1, 0] = 0.0
    
    gain_weight = numpy.ones(gainshape)
    gain_time = utimes
    gain_frequency = ufrequency
    gain_residual = numpy.zeros([ntimes, nfrequency, nrec, nrec])
    
    gt = GainTable(gain=gain, time=gain_time, weight=gain_weight, residual=gain_residual, frequency=gain_frequency,
                   receptor_frame=receptor_frame)
    
    assert type(gt) is GainTable, "gt is not a GainTable: %r" % gt
    
    return gt


def apply_gaintable(vis: BlockVisibility, gt: GainTable, inverse=False) -> BlockVisibility:
    """Apply a gain table to a block visibility
    
    The corrected visibility is::
    
        V_corrected = {g_i * g_j^*}^-1 V_obs
        
    If the visibility data are polarised e.g. polarisation_frame("linear") then the inverse operator
    represents an actual inverse of the gains.
    
    :param vis: Visibility to have gains applied
    :param gt: Gaintable to be applied
    :param inverse: Apply the inverse (default=False)
    :returns: input vis with gains applied
    
    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis
    assert type(gt) is GainTable, "gt is not a GainTable: %r" % gt
    
    if inverse:
        log.info('apply_gaintable: Apply inverse gaintable')
    else:
        log.info('apply_gaintable: Apply gaintable')
    
    for chunk, rows in enumerate(vis_timeslice_iter(vis)):
        vistime = numpy.average(vis.time[rows])
        integration_time = numpy.average(vis.integration_time[rows])
        gaintable_rows = abs(gt.time - vistime) < integration_time / 2.0
        
        # Lookup the gain for this set of visibilities
        gain = gt.data['gain'][gaintable_rows]
        gwt = gt.data['weight'][gaintable_rows]
        
        # The shape of the mueller matrix is
        ntimes, nant, nchan, nrec, _ = gain.shape
        # muellershape = (ntimes, nant, nant, nchan, nrec * nrec, nrec * nrec)
        # mueller = numpy.zeros(muellershape, dtype='complex')
        
        original = vis.vis[rows]
        applied = copy.deepcopy(original)
        for time in range(ntimes):
            for a1 in range(vis.nants - 1):
                for a2 in range(a1 + 1, vis.nants):
                    for chan in range(nchan):
                        mueller = numpy.kron(gain[time, a1, chan, :, :], numpy.conjugate(gain[time, a2, chan, :, :]))
                        if inverse:
                            mueller = numpy.linalg.inv(mueller)
                        
                        applied[time, a2, a1, chan, :] = numpy.matmul(mueller, original[time, a2, a1, chan, :])
        
        vis.data['vis'][rows] = applied
    return vis


def solve_gaintable(vis: BlockVisibility, modelvis: BlockVisibility, phase_only=True, niter=30, tol=1e-8,
                    crosspol=False) -> GainTable:
    """Solve a gain table to a block visibility
    
    :param vis: BlockVisibility containing the observed data
    :param modelvis: BlockVisibility containing the visibility predicted by a model
    :param phase_only: Solve only for the phases (default=True)
    :param niter: Number of iterations (default 30)
    :param tol: Iteration stops when the fractional change in the gain solution is below this tolerance
    :param crosspol: Do solutions including cross polarisations i.e. XY, YX or RL, LR
    :returns: GainTable containing solution
    
    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis
    assert type(modelvis) is BlockVisibility, "modelvis is not a BlockVisibility: %r" % vis
    
    if phase_only:
        log.info('solve_gaintable: Solving for phase only')
    else:
        log.info('solve_gaintable: Solving for complex gain')
    
    gt = create_gaintable_from_blockvisibility(vis)
    
    for chunk, rows in enumerate(vis_timeslice_iter(vis)):
        # Form the point source equivalent visibility
        X = numpy.zeros_like(vis.vis[rows])
        Xwt = numpy.abs(modelvis.vis[rows]) ** 2 * modelvis.weight[rows]
        mask = Xwt > 0.0
        X[mask] = vis.vis[rows][mask] / modelvis.vis[rows][mask]
        
        # Now average over time, chan. The axes of X are time, antenna2, antenna1, chan, pol
        
        Xave = numpy.average(X * Xwt, axis=0)
        XwtAve = numpy.average(Xwt, axis=0)
        
        mask = XwtAve > 0.0
        Xave[mask] = Xave[mask] / XwtAve[mask]
        
        gainshape = gt.data['gain'][chunk, ...].shape
        if vis.polarisation_frame.npol > 1:
            if crosspol:
                gt.data['gain'][chunk, ...], gt.data['weight'][chunk, ...], gt.data['residual'][chunk, ...] = \
                    solve_antenna_gains_itsubs_matrix(gainshape, Xave, XwtAve, phase_only=phase_only, niter=niter,
                                                      tol=tol)
            else:
                gt.data['gain'][chunk, ...], gt.data['weight'][chunk, ...], gt.data['residual'][chunk, ...] = \
                    solve_antenna_gains_itsubs_vector(gainshape, Xave, XwtAve, phase_only=phase_only, niter=niter,
                                                      tol=tol)
        
        else:
            gt.data['gain'][chunk, ...], gt.data['weight'][chunk, ...], gt.data['residual'][chunk, ...] = \
                solve_antenna_gains_itsubs_scalar(gainshape, Xave, XwtAve, phase_only=phase_only, niter=niter,
                                                  tol=tol)
    
    assert type(gt) is GainTable, "gt is not a GainTable: %r" % gt
    
    return gt


def solve_antenna_gains_itsubs_scalar(gainshape, X, Xwt, niter=30, tol=1e-8, phase_only=True, refant=0):
    """Solve for the antenna gains

    X(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))

    This uses an iterative substitution algorithm due to Larry D'Addario c 1980'ish. Used
    in the original VLA Dec-10 Antsol.

    :param gainshape: Shape of output gains
    :param X: Equivalent point source visibility[nants, nants, ...]
    :param Xwt: Equivalent point source weight [nants, nants, ...]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :param phase_only: Do solution for only the phase? (default True)
    :param refant: Reference antenna for phase (default=0.0)
    :returns: gain [nants, ...], weight [nants, ...]
    """
    
    nants = X.shape[0]
    for ant1 in range(nants):
        X[ant1, ant1, ...] = 0.0
        Xwt[ant1, ant1, ...] = 0.0
        for ant2 in range(ant1 + 1, nants):
            X[ant1, ant2, ...] = numpy.conjugate(X[ant2, ant1, ...])
            Xwt[ant1, ant2, ...] = Xwt[ant2, ant1, ...]
    
    gain = numpy.ones(shape=gainshape, dtype=X.dtype)
    gwt = numpy.zeros(shape=gainshape, dtype=Xwt.dtype)
    for iter in range(niter):
        gainLast = gain
        gain, gwt = gain_substitution_scalar(gain, X, Xwt)
        if phase_only:
            mask = numpy.abs(gain) > 0.0
            gain[mask] = gain[mask] / numpy.abs(gain[mask])
        gain *= numpy.conjugate(gain[refant, ...]) / numpy.abs(gain[refant, ...])
        gain = 0.5 * (gain + gainLast)
        change = numpy.max(numpy.abs(gain - gainLast))
        if change < tol:
            return gain, gwt, solution_residual_scalar(gain, X, Xwt)
    
    return gain, gwt, solution_residual_scalar(gain, X, Xwt)


def gain_substitution_scalar(gain, X, Xwt):
    nants, nchan, nrec, _ = gain.shape
    newgain = numpy.ones_like(gain, dtype='complex')
    gwt = numpy.zeros_like(gain, dtype='float')
    
    # We are going to work with Jones 2x2 matrix formalism so everything has to be
    # converted to that format
    X = X.reshape(nants, nants, nchan, nrec, nrec)
    Xwt = Xwt.reshape(nants, nants, nchan, nrec, nrec)
    
    for ant1 in range(nants):
        for chan in range(nchan):
            # Loop over e.g. 'RR', 'LL, or 'XX', 'YY' ignoring cross terms
            top = numpy.sum(X[:, ant1, chan, 0, 0] \
                            * gain[:, chan, 0, 0] * Xwt[:, ant1, chan, 0, 0], axis=0)
            bot = numpy.sum((gain[:, chan, 0, 0] * numpy.conjugate(gain[:, chan, 0, 0])
                             * Xwt[:, ant1, chan, 0, 0]).real, axis=0)
            
            if bot > 0.0:
                newgain[ant1, chan, 0, 0] = top / bot
                gwt[ant1, chan, 0, 0] = bot
            else:
                newgain[ant1, chan, 0, 0] = 0.0
                gwt[ant1, chan, 0, 0] = 0.0
    return newgain, gwt


def solve_antenna_gains_itsubs_vector(gainshape, X, Xwt, niter=30, tol=1e-8, phase_only=True, refant=0):
    """Solve for the antenna gains using full matrix expressions

    X(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))

    See Appendix D, section D.1 in:
    
    J. P. Hamaker, “Understanding radio polarimetry - IV. The full-coherency analogue of
    scalar self-calibration: Self-alignment, dynamic range and polarimetric fidelity,” Astronomy
    and Astrophysics Supplement Series, vol. 143, no. 3, pp. 515–534, May 2000.

    :param gainshape: Shape of output gains
    :param X: Equivalent point source visibility[nants, nants, ...]
    :param Xwt: Equivalent point source weight [nants, nants, ...]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :param phase_only: Do solution for only the phase? (default True)
    :param refant: Reference antenna for phase (default=0.0)
    :returns: gain [nants, ...], weight [nants, ...]
    """
    
    nants, _, nchan, npol = X.shape
    for ant1 in range(nants):
        X[ant1, ant1, ...] = 0.0
        Xwt[ant1, ant1, ...] = 0.0
        for ant2 in range(ant1 + 1, nants):
            X[ant1, ant2, ...] = numpy.conjugate(X[ant2, ant1, ...])
            Xwt[ant1, ant2, ...] = Xwt[ant2, ant1, ...]
    
    nrec = gainshape[-1]
    assert npol == nrec * nrec
    
    gain = numpy.ones(shape=gainshape, dtype=X.dtype)
    gain[..., 0, 1] = 0.0
    gain[..., 1, 0] = 0.0
    gwt = numpy.zeros(shape=gainshape, dtype=Xwt.dtype)
    
    for iter in range(niter):
        gainLast = gain
        gain, gwt = gain_substitution_vector(gain, X, Xwt)
        for rec in [0, 1]:
            gain[..., rec, 1 - rec] = 0.0
            if phase_only:
                gain[..., rec, rec] = gain[..., rec, rec] / numpy.abs(gain[..., rec, rec])
            gain[..., rec, rec] *= numpy.conjugate(gain[refant, ..., rec, rec]) / numpy.abs(gain[refant, ..., rec, rec])
        gain = 0.5 * (gain + gainLast)
        change = numpy.max(numpy.abs(gain - gainLast))
        if change < tol:
            return gain, gwt, solution_residual_vector(gain, X, Xwt)
    
    return gain, gwt, solution_residual_vector(gain, X, Xwt)


def gain_substitution_vector(gain, X, Xwt):
    nants, nchan, nrec, _ = gain.shape
    newgain = numpy.ones_like(gain, dtype='complex')
    if nrec > 0:
        newgain[..., 0, 1] = 0.0
        newgain[..., 1, 0] = 0.0
    
    gwt = numpy.zeros_like(gain, dtype='float')
    
    # We are going to work with Jones 2x2 matrix formalism so everything has to be
    # converted to that format
    X = X.reshape(nants, nants, nchan, nrec, nrec)
    Xwt = Xwt.reshape(nants, nants, nchan, nrec, nrec)
    
    if nrec > 0:
        gain[..., 0, 1] = 0.0
        gain[..., 1, 0] = 0.0
    
    for ant1 in range(nants):
        for chan in range(nchan):
            # Loop over e.g. 'RR', 'LL, or 'XX', 'YY' ignoring cross terms
            for rec in range(nrec):
                top = numpy.sum(X[:, ant1, chan, rec, rec] \
                                * gain[:, chan, rec, rec] * Xwt[:, ant1, chan, rec, rec], axis=0)
                bot = numpy.sum((gain[:, chan, rec, rec] * numpy.conjugate(gain[:, chan, rec, rec])
                                 * Xwt[:, ant1, chan, rec, rec]).real, axis=0)
                
                if bot > 0.0:
                    newgain[ant1, chan, rec, rec] = top / bot
                    gwt[ant1, chan, rec, rec] = bot
                else:
                    newgain[ant1, chan, rec, rec] = 0.0
                    gwt[ant1, chan, rec, rec] = 0.0
    
    return newgain, gwt


def solve_antenna_gains_itsubs_matrix(gainshape, X, Xwt, niter=30, tol=1e-8, phase_only=True, refant=0):
    """Solve for the antenna gains using full matrix expressions

    X(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))

    See Appendix D, section D.1 in:

    J. P. Hamaker, “Understanding radio polarimetry - IV. The full-coherency analogue of
    scalar self-calibration: Self-alignment, dynamic range and polarimetric fidelity,” Astronomy
    and Astrophysics Supplement Series, vol. 143, no. 3, pp. 515–534, May 2000.

    :param gainshape: Shape of gaintable
    :param X: Equivalent point source visibility[nants, nants, ...]
    :param Xwt: Equivalent point source weight [nants, nants, ...]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :param phase_only: Do solution for only the phase? (default True)
    :param refant: Reference antenna for phase (default=0.0)
    :returns: gain [nants, ...], weight [nants, ...]
    """
    
    nants, _, nchan, npol = X.shape
    for ant1 in range(nants):
        X[ant1, ant1, ...] = 0.0
        Xwt[ant1, ant1, ...] = 0.0
        for ant2 in range(ant1 + 1, nants):
            X[ant1, ant2, ...] = numpy.conjugate(X[ant2, ant1, ...])
            Xwt[ant1, ant2, ...] = Xwt[ant2, ant1, ...]
    
    nrec = gainshape[-1]
    assert npol == nrec * nrec
    
    gain = numpy.ones(shape=gainshape, dtype=X.dtype)
    gain[..., 0, 1] = 0.0
    gain[..., 1, 0] = 0.0
    gwt = numpy.zeros(shape=gainshape, dtype=Xwt.dtype)
    
    for iter in range(niter):
        gainLast = gain
        gain, gwt = gain_substitution_vector(gain, X, Xwt)
        if phase_only:
            gain = gain / numpy.abs(gain)
        # mask = numpy.abs(gain) > 0.0
        # gain[mask] *= numpy.conjugate(gain[mask][refant, ...]) / numpy.abs(gain[refant, ...][mask])
        gain = 0.5 * (gain + gainLast)
        change = numpy.max(numpy.abs(gain - gainLast))
        if change < tol:
            return gain, gwt, solution_residual_vector(gain, X, Xwt)
    
    return gain, gwt, solution_residual_vector(gain, X, Xwt)


def gain_substitution_matrix(gain, X, Xwt):
    nants, nchan, nrec, _ = gain.shape
    newgain = numpy.ones_like(gain, dtype='complex')
    gwt = numpy.zeros_like(gain, dtype='float')
    
    # We are going to work with Jones 2x2 matrix formalism so everything has to be
    # converted to that format
    X = X.reshape(nants, nants, nchan, nrec, nrec)
    Xwt = Xwt.reshape(nants, nants, nchan, nrec, nrec)
    
    # Write these loops out explicitly. Derivation of these vector equations is tedious but they are
    # structurally identical to the scalar case with the following changes
    # Vis -> 2x2 coherency vector, g-> 2x2 Jones matrix, *-> matmul, conjugate->Hermitean transpose (.H)
    for ant1 in range(nants):
        for chan in range(nchan):
            top = 0.0
            bot = 0.0
            for ant2 in range(nants):
                # Xmat = numpy.matrix(X[ant2, ant1, chan]).reshape([2,2])
                # Xwtmat = numpy.matrix(Xwt[ant2, ant1, chan]).reshape([2,2])
                # g = numpy.matrix(gain[ant2, chan])
                # top += numpy.matmul(numpy.matmul(Xmat, g), Xwtmat)
                # bot += numpy.matmul(numpy.matmul(g, g.H), Xwtmat)
                top += numpy.dot(numpy.dot(X[ant2, ant1, chan], gain[ant2, chan]), Xwt[ant2, ant1, chan])
                bot += numpy.dot(numpy.dot(gain[ant2, chan], numpy.conjugate(gain[ant2, chan])),
                                 Xwt[ant2, ant1, chan]).real
            
            newgain[ant1, chan] = numpy.dot(numpy.linalg.inv(numpy.matrix(bot)), top)
            gwt[ant1, chan] = bot.real
    return newgain, gwt


def solution_residual_scalar(gain, X, Xwt):
    """Calculate residual across all baselines of gain for point source equivalent visibilities
    
    :param gain: gain [nant, ...]
    :param X: Point source equivalent visibility [nant, ...]
    :param Xwt: Point source equivalent weight [nant, ...]
    :returns: residual[...]
    """
    
    nants, nchan, nrec, _ = gain.shape
    X = X.reshape(nants, nants, nchan, nrec, nrec)
    
    Xwt = Xwt.reshape(nants, nants, nchan, nrec, nrec)
    
    residual = numpy.zeros([nchan, nrec, nrec])
    sumwt = numpy.zeros([nchan, nrec, nrec])
    
    for ant1 in range(nants):
        for ant2 in range(nants):
            for chan in range(nchan):
                error = X[ant2, ant1, chan, 0, 0] - \
                        gain[ant1, chan, 0, 0] * numpy.conjugate(gain[ant2, chan, 0, 0])
                residual += (error * Xwt[ant2, ant1, chan, 0, 0] * numpy.conjugate(error)).real
                sumwt += Xwt[ant2, ant1, chan, 0, 0]
    
    residual = numpy.sqrt(residual / sumwt)
    return residual


def solution_residual_vector(gain, X, Xwt):
    """Calculate residual across all baselines of gain for point source equivalent visibilities
    
    Vector case i.e. off-diagonals of gains are zero

    :param gain: gain [nant, ...]
    :param X: Point source equivalent visibility [nant, ...]
    :param Xwt: Point source equivalent weight [nant, ...]
    :returns: residual[...]
    """
    
    nants, nchan, nrec, _ = gain.shape
    X = X.reshape(nants, nants, nchan, nrec, nrec)
    X[..., 1, 0] = 0.0
    X[..., 0, 1] = 0.0
    
    Xwt = Xwt.reshape(nants, nants, nchan, nrec, nrec)
    Xwt[..., 1, 0] = 0.0
    Xwt[..., 0, 1] = 0.0
    
    residual = numpy.zeros([nchan, nrec, nrec])
    sumwt = numpy.zeros([nchan, nrec, nrec])
    
    for ant1 in range(nants):
        for ant2 in range(nants):
            for chan in range(nchan):
                for rec in range(nrec):
                    error = X[ant2, ant1, chan, rec, rec] - \
                            gain[ant1, chan, rec, rec] * numpy.conjugate(gain[ant2, chan, rec, rec])
                    residual += (error * Xwt[ant2, ant1, chan, rec, rec] * numpy.conjugate(error)).real
                    sumwt += Xwt[ant2, ant1, chan, rec, rec]
    
    residual = numpy.sqrt(residual / sumwt)
    return residual


def solution_residual_matrix(gain, X, Xwt):
    """Calculate residual across all baselines of gain for point source equivalent visibilities

    :param gain: gain [nant, ...]
    :param X: Point source equivalent visibility [nant, ...]
    :param Xwt: Point source equivalent weight [nant, ...]
    :returns: residual[...]
    """
    
    nants, nchan, nrec, _ = gain.shape
    X = X.reshape(nants, nants, nchan, nrec, nrec)
    Xwt = Xwt.reshape(nants, nants, nchan, nrec, nrec)
    
    residual = numpy.zeros([nchan, nrec, nrec])
    sumwt = numpy.zeros([nchan, nrec, nrec])
    
    for ant1 in range(nants):
        for ant2 in range(nants):
            for chan in range(nchan):
                error = X[ant2, ant1, chan] - numpy.dot(gain[ant1, chan], numpy.conjugate(gain[ant2, chan]))
                residual += (numpy.dot(numpy.dot(error, Xwt[ant2, ant1, chan]), numpy.conjugate(error))).real
                sumwt += Xwt[ant2, ant1, chan]
    
    residual = numpy.sqrt(residual / sumwt)
    return residual
