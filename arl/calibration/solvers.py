import numpy
import logging
import collections

from arl.data.data_models import BlockVisibility, GainTable, BlockVisibility, Skycomponent
from arl.visibility.iterators import vis_timeslice_iter
from arl.visibility.operations import copy_visibility
from arl.calibration.operations import create_gaintable_from_blockvisibility
from arl.fourier_transforms.ftprocessor_base import predict_skycomponent_blockvisibility
from arl.calibration.operations import apply_gaintable

log = logging.getLogger(__name__)

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
        
        x, xwt = remove_model(vis.vis[rows], vis.weight[rows], modelvis.vis[rows],
                              isscalar=vis.polarisation_frame.npol == 1, crosspol=crosspol)
        
        # Now average over time, chan. The axes of x are time, antenna2, antenna1, chan, pol
        
        xave = numpy.average(x, axis=0)
        xwtAve = numpy.average(xwt, axis=0)
        
        mask = xwtAve <= 0.0
        xave[mask] = 0.0
        
        gainshape = gt.data['gain'][chunk, ...].shape
        if vis.polarisation_frame.npol > 1:
            if crosspol:
                gt.data['gain'][chunk, ...], gt.data['weight'][chunk, ...], gt.data['residual'][chunk, ...] = \
                    solve_antenna_gains_itsubs_matrix(gainshape, xave, xwtAve, phase_only=phase_only, niter=niter,
                                                      tol=tol)
            else:
                gt.data['gain'][chunk, ...], gt.data['weight'][chunk, ...], gt.data['residual'][chunk, ...] = \
                    solve_antenna_gains_itsubs_vector(gainshape, xave, xwtAve, phase_only=phase_only, niter=niter,
                                                      tol=tol)
        
        else:
            gt.data['gain'][chunk, ...], gt.data['weight'][chunk, ...], gt.data['residual'][chunk, ...] = \
                solve_antenna_gains_itsubs_scalar(gainshape, xave, xwtAve, phase_only=phase_only, niter=niter,
                                                  tol=tol)
    
    assert type(gt) is GainTable, "gt is not a GainTable: %r" % gt
    
    return gt


def remove_model(vis, weight, modelvis, isscalar, crosspol):
    # Form the point source equivalent visibility
 
    # Different for scalar and vector/matrix cases
    
    if isscalar:
        # Scalar case is straightforward
        x = numpy.zeros_like(vis)
        xwt = numpy.abs(modelvis) ** 2 * weight
        mask = xwt > 0.0
        x[mask] = vis[mask] / modelvis[mask]
    else:
        nrows, nants, _, nchan, npol = vis.shape
        nrec = 2
        assert nrec * nrec == npol
        xshape = (nrows, nants, nants, nchan, nrec, nrec)
        x = numpy.zeros(xshape, dtype='complex')
        xwt = numpy.zeros(xshape)
        for row in range(nrows):
            for ant1 in range(nants):
                for ant2 in range(ant1+1, nants):
                    for chan in range(nchan):
                        ovis = numpy.matrix(vis[row, ant2, ant1, chan].reshape([2,2]))
                        mvis = numpy.matrix(modelvis[row, ant2, ant1, chan].reshape([2,2]))
                        wt = numpy.matrix(weight[row, ant2, ant1, chan].reshape([2,2]))
                        x[row, ant2, ant1, chan] = numpy.matmul(numpy.linalg.inv(mvis), ovis)
                        xwt[row, ant2, ant1, chan] = numpy.dot(mvis, numpy.multiply(wt, mvis.H)).real
                        
    return x, xwt


def solve_antenna_gains_itsubs_scalar(gainshape, x, xwt, niter=30, tol=1e-8, phase_only=True, refant=0):
    """Solve for the antenna gains

    x(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))

    This uses an iterative substitution algorithm due to Larry D'Addario c 1980'ish. Used
    in the original VLA Dec-10 Antsol.

    :param gainshape: Shape of output gains
    :param x: Equivalent point source visibility[nants, nants, ...]
    :param xwt: Equivalent point source weight [nants, nants, ...]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :param phase_only: Do solution for only the phase? (default True)
    :param refant: Reference antenna for phase (default=0.0)
    :returns: gain [nants, ...], weight [nants, ...]
    """
    
    nants = x.shape[0]
    for ant1 in range(nants):
        x[ant1, ant1, ...] = 0.0
        xwt[ant1, ant1, ...] = 0.0
        for ant2 in range(ant1 + 1, nants):
            x[ant1, ant2, ...] = numpy.conjugate(x[ant2, ant1, ...])
            xwt[ant1, ant2, ...] = xwt[ant2, ant1, ...]
    
    gain = numpy.ones(shape=gainshape, dtype=x.dtype)
    gwt = numpy.zeros(shape=gainshape, dtype=xwt.dtype)
    for iter in range(niter):
        gainLast = gain
        gain, gwt = gain_substitution_scalar(gain, x, xwt)
        if phase_only:
            mask = numpy.abs(gain) > 0.0
            gain[mask] = gain[mask] / numpy.abs(gain[mask])
        gain *= numpy.conjugate(gain[refant, ...]) / numpy.abs(gain[refant, ...])
        gain = 0.5 * (gain + gainLast)
        change = numpy.max(numpy.abs(gain - gainLast))
        if change < tol:
            return gain, gwt, solution_residual_scalar(gain, x, xwt)
    
    return gain, gwt, solution_residual_scalar(gain, x, xwt)


def gain_substitution_scalar(gain, x, xwt):
    nants, nchan, nrec, _ = gain.shape
    newgain = numpy.ones_like(gain, dtype='complex')
    gwt = numpy.zeros_like(gain, dtype='float')
    
    # We are going to work with Jones 2x2 matrix formalism so everything has to be
    # converted to that format
    x = x.reshape(nants, nants, nchan, nrec, nrec)
    xwt = xwt.reshape(nants, nants, nchan, nrec, nrec)
    
    for ant1 in range(nants):
        for chan in range(nchan):
            # Loop over e.g. 'RR', 'LL, or 'xx', 'YY' ignoring cross terms
            top = numpy.sum(x[:, ant1, chan, 0, 0] \
                            * gain[:, chan, 0, 0] * xwt[:, ant1, chan, 0, 0], axis=0)
            bot = numpy.sum((gain[:, chan, 0, 0] * numpy.conjugate(gain[:, chan, 0, 0])
                             * xwt[:, ant1, chan, 0, 0]).real, axis=0)
            
            if bot > 0.0:
                newgain[ant1, chan, 0, 0] = top / bot
                gwt[ant1, chan, 0, 0] = bot
            else:
                newgain[ant1, chan, 0, 0] = 0.0
                gwt[ant1, chan, 0, 0] = 0.0
    return newgain, gwt


def solve_antenna_gains_itsubs_vector(gainshape, x, xwt, niter=30, tol=1e-8, phase_only=True, refant=0):
    """Solve for the antenna gains using full matrix expressions

    x(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))

    See Appendix D, section D.1 in:
    
    J. P. Hamaker, “Understanding radio polarimetry - IV. The full-coherency analogue of
    scalar self-calibration: Self-alignment, dynamic range and polarimetric fidelity,” Astronomy
    and Astrophysics Supplement Series, vol. 143, no. 3, pp. 515–534, May 2000.

    :param gainshape: Shape of output gains
    :param x: Equivalent point source visibility[nants, nants, ...]
    :param xwt: Equivalent point source weight [nants, nants, ...]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :param phase_only: Do solution for only the phase? (default True)
    :param refant: Reference antenna for phase (default=0.0)
    :returns: gain [nants, ...], weight [nants, ...]
    """
    
    nants, _, nchan, nrec, _ = x.shape
    for ant1 in range(nants):
        x[ant1, ant1, ...] = 0.0
        xwt[ant1, ant1, ...] = 0.0
        for ant2 in range(ant1 + 1, nants):
            x[ant1, ant2, ...] = numpy.conjugate(x[ant2, ant1, ...])
            xwt[ant1, ant2, ...] = xwt[ant2, ant1, ...]
    
    gain = numpy.ones(shape=gainshape, dtype=x.dtype)
    gain[..., 0, 1] = 0.0
    gain[..., 1, 0] = 0.0
    gwt = numpy.zeros(shape=gainshape, dtype=xwt.dtype)
    
    for iter in range(niter):
        gainLast = gain
        gain, gwt = gain_substitution_vector(gain, x, xwt)
        for rec in [0, 1]:
            gain[..., rec, 1 - rec] = 0.0
            if phase_only:
                gain[..., rec, rec] = gain[..., rec, rec] / numpy.abs(gain[..., rec, rec])
            gain[..., rec, rec] *= numpy.conjugate(gain[refant, ..., rec, rec]) / numpy.abs(gain[refant, ..., rec, rec])
        change = numpy.max(numpy.abs(gain - gainLast))
        gain = 0.5 * (gain + gainLast)
        if change < tol:
            return gain, gwt, solution_residual_vector(gain, x, xwt)
    
    return gain, gwt, solution_residual_vector(gain, x, xwt)


def gain_substitution_vector(gain, x, xwt):
    nants, nchan, nrec, _ = gain.shape
    newgain = numpy.ones_like(gain, dtype='complex')
    if nrec > 0:
        newgain[..., 0, 1] = 0.0
        newgain[..., 1, 0] = 0.0
    
    gwt = numpy.zeros_like(gain, dtype='float')
    
    # We are going to work with Jones 2x2 matrix formalism so everything has to be
    # converted to that format
    x = x.reshape(nants, nants, nchan, nrec, nrec)
    xwt = xwt.reshape(nants, nants, nchan, nrec, nrec)
    
    if nrec > 0:
        gain[..., 0, 1] = 0.0
        gain[..., 1, 0] = 0.0
    
    for ant1 in range(nants):
        for chan in range(nchan):
            # Loop over e.g. 'RR', 'LL, or 'xx', 'YY' ignoring cross terms
            for rec in range(nrec):
                top = numpy.sum(x[:, ant1, chan, rec, rec] \
                                * gain[:, chan, rec, rec] * xwt[:, ant1, chan, rec, rec], axis=0)
                bot = numpy.sum((gain[:, chan, rec, rec] * numpy.conjugate(gain[:, chan, rec, rec])
                                 * xwt[:, ant1, chan, rec, rec]).real, axis=0)
                
                if bot > 0.0:
                    newgain[ant1, chan, rec, rec] = top / bot
                    gwt[ant1, chan, rec, rec] = bot
                else:
                    newgain[ant1, chan, rec, rec] = 0.0
                    gwt[ant1, chan, rec, rec] = 0.0
    
    return newgain, gwt


def solve_antenna_gains_itsubs_matrix(gainshape, x, xwt, niter=30, tol=1e-8, phase_only=True, refant=0):
    """Solve for the antenna gains using full matrix expressions

    x(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))

    See Appendix D, section D.1 in:

    J. P. Hamaker, “Understanding radio polarimetry - IV. The full-coherency analogue of
    scalar self-calibration: Self-alignment, dynamic range and polarimetric fidelity,” Astronomy
    and Astrophysics Supplement Series, vol. 143, no. 3, pp. 515–534, May 2000.

    :param gainshape: Shape of gaintable
    :param x: Equivalent point source visibility[nants, nants, ...]
    :param xwt: Equivalent point source weight [nants, nants, ...]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :param phase_only: Do solution for only the phase? (default True)
    :param refant: Reference antenna for phase (default=0.0)
    :returns: gain [nants, ...], weight [nants, ...]
    """
    
    nants, _, nchan, nrec, _ = x.shape
    for ant1 in range(nants):
        x[ant1, ant1, ...] = 0.0
        xwt[ant1, ant1, ...] = 0.0
        for ant2 in range(ant1 + 1, nants):
            x[ant1, ant2, ...] = numpy.conjugate(x[ant2, ant1, ...])
            xwt[ant1, ant2, ...] = xwt[ant2, ant1, ...]
        
    gain = numpy.ones(shape=gainshape, dtype=x.dtype)
    gain[..., 0, 1] = 0.0
    gain[..., 1, 0] = 0.0
    gwt = numpy.zeros(shape=gainshape, dtype=xwt.dtype)
    
    for iter in range(niter):
        gainLast = gain
        gain, gwt = gain_substitution_matrix(gain, x, xwt)
        if phase_only:
            gain = gain / numpy.abs(gain)
        change = numpy.max(numpy.abs(gain - gainLast))
        gain = 0.5 * (gain + gainLast)
        if change < tol:
            return gain, gwt, solution_residual_matrix(gain, x, xwt)
    
    return gain, gwt, solution_residual_matrix(gain, x, xwt)


def gain_substitution_matrix(gain, x, xwt):
    nants, nchan, nrec, _ = gain.shape
    newgain = numpy.ones_like(gain, dtype='complex')
    gwt = numpy.zeros_like(gain, dtype='float')
    
    # We are going to work with Jones 2x2 matrix formalism so everything has to be
    # converted to that format
    x = x.reshape(nants, nants, nchan, nrec, nrec)
    xwt = xwt.reshape(nants, nants, nchan, nrec, nrec)
    
    # Write these loops out explicitly. Derivation of these vector equations is tedious but they are
    # structurally identical to the scalar case with the following changes
    # Vis -> 2x2 coherency vector, g-> 2x2 Jones matrix, *-> matmul, conjugate->Hermitean transpose (.H)
    for ant1 in range(nants):
        for chan in range(nchan):
            top = 0.0
            bot = 0.0
            for ant2 in range(nants):
                if ant1 != ant2:
                    xmat = x[ant2, ant1, chan]
                    xwtmat = xwt[ant2, ant1, chan]
                    g2 = gain[ant2, chan]
                    top += xmat * xwtmat * g2
                    bot += numpy.conjugate(g2) * xwtmat * g2
            newgain[ant1, chan][bot>0.0] = top[bot>0.0] / bot[bot>0.0]
            newgain[ant1, chan][bot<=0.0] = 0.0
            gwt[ant1, chan] = bot.real
    return newgain, gwt


def solution_residual_scalar(gain, x, xwt):
    """Calculate residual across all baselines of gain for point source equivalent visibilities
    
    :param gain: gain [nant, ...]
    :param x: Point source equivalent visibility [nant, ...]
    :param xwt: Point source equivalent weight [nant, ...]
    :returns: residual[...]
    """
    
    nants, nchan, nrec, _ = gain.shape
    x = x.reshape(nants, nants, nchan, nrec, nrec)
    
    xwt = xwt.reshape(nants, nants, nchan, nrec, nrec)
    
    residual = numpy.zeros([nchan, nrec, nrec])
    sumwt = numpy.zeros([nchan, nrec, nrec])
    
    for ant1 in range(nants):
        for ant2 in range(nants):
            for chan in range(nchan):
                error = x[ant2, ant1, chan, 0, 0] - \
                        gain[ant1, chan, 0, 0] * numpy.conjugate(gain[ant2, chan, 0, 0])
                residual += (error * xwt[ant2, ant1, chan, 0, 0] * numpy.conjugate(error)).real
                sumwt += xwt[ant2, ant1, chan, 0, 0]
    
    residual = numpy.sqrt(residual / sumwt)
    return residual


def solution_residual_vector(gain, x, xwt):
    """Calculate residual across all baselines of gain for point source equivalent visibilities
    
    Vector case i.e. off-diagonals of gains are zero

    :param gain: gain [nant, ...]
    :param x: Point source equivalent visibility [nant, ...]
    :param xwt: Point source equivalent weight [nant, ...]
    :returns: residual[...]
    """
    
    nants, nchan, nrec, _ = gain.shape
    x = x.reshape(nants, nants, nchan, nrec, nrec)
    x[..., 1, 0] = 0.0
    x[..., 0, 1] = 0.0
    
    xwt = xwt.reshape(nants, nants, nchan, nrec, nrec)
    xwt[..., 1, 0] = 0.0
    xwt[..., 0, 1] = 0.0
    
    residual = numpy.zeros([nchan, nrec, nrec])
    sumwt = numpy.zeros([nchan, nrec, nrec])
    
    for ant1 in range(nants):
        for ant2 in range(nants):
            for chan in range(nchan):
                for rec in range(nrec):
                    error = x[ant2, ant1, chan, rec, rec] - \
                            gain[ant1, chan, rec, rec] * numpy.conjugate(gain[ant2, chan, rec, rec])
                    residual += (error * xwt[ant2, ant1, chan, rec, rec] * numpy.conjugate(error)).real
                    sumwt += xwt[ant2, ant1, chan, rec, rec]
    
    residual = numpy.sqrt(residual / sumwt)
    return residual


def solution_residual_matrix(gain, x, xwt):
    """Calculate residual across all baselines of gain for point source equivalent visibilities

    :param gain: gain [nant, ...]
    :param x: Point source equivalent visibility [nant, ...]
    :param xwt: Point source equivalent weight [nant, ...]
    :returns: residual[...]
    """
    
    nants, nchan, nrec, _ = gain.shape
    x = x.reshape(nants, nants, nchan, nrec, nrec)
    xwt = xwt.reshape(nants, nants, nchan, nrec, nrec)
    
    residual = numpy.zeros([nchan, nrec, nrec])
    sumwt = numpy.zeros([nchan, nrec, nrec])


    # This is written out in long winded form but should be optimised for
    # production code!
    for ant1 in range(nants):
        for ant2 in range(nants):
            for chan in range(nchan):
                for rec1 in range(nrec):
                    for rec2 in range(nrec):
                        error = x[ant2, ant1, chan, rec2, rec1] - \
                                gain[ant1, chan, rec2, rec1] * numpy.conjugate(gain[ant2, chan, rec2, rec1])
                        residual[chan, rec2, rec1] += (error * xwt[ant2, ant1, chan, rec2, rec1] * numpy.conjugate(
                            error)).real
                        sumwt[chan, rec2, rec1] += xwt[ant2, ant1, chan, rec2, rec1]

    residual[sumwt>0.0] = numpy.sqrt(residual[sumwt>0.0] / sumwt[sumwt>0.0])
    residual[sumwt <= 0.0] = 0.0
    return residual

def peel_skycomponent_blockvisibility(vis: BlockVisibility, sc: Skycomponent, remove=True) -> \
        BlockVisibility:
    """ Peel a collection of component.
    
    Sequentially solve the gain towards each Skycomponent and optionally remove from the visibility.

    :param params:
    :param vis: Visibility to be processed
    :param sc: Skycomponent or list of Skycomponents
    :returns: subtracted visibility and list of GainTables
    """
    # TODO: Implement peeling
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis

    if not isinstance(sc, collections.Iterable):
        sc = [sc]

    gtlist = []
    for comp in sc:
        assert comp.shape == 'Point', "Cannot handle shape %s" % comp.shape
        
        modelvis = copy_visibility(vis)
        modelvis = predict_skycomponent_blockvisibility(modelvis, comp)
        gt = solve_gaintable(vis, modelvis)
        modelvis = apply_gaintable(modelvis, gt)
        if remove:
            vis.data -= modelvis.data
        gtlist.append(gt)
        
    return vis, gtlist