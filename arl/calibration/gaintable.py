# Tim Cornwell <realtimcornwell@gmail.com>
#
""" Visibility operations

"""

import copy
from arl.data.data_models import GainTable, BlockVisibility
from arl.fourier_transforms.ftprocessor_params import *
from arl.visibility.operations import create_blockvisibility_from_rows
from arl.visibility.iterators import vis_timeslice_iter
from arl.util.coordinate_support import *

log = logging.getLogger(__name__)


def gaintable_summary(gt):
    """Return string summarizing the Gaintable

    """
    return "%s rows, %.3f GB" % (gt.data.shape, gt.size())

def create_gaintable_from_blockvisibility(vis: BlockVisibility, time_width: float=None, frequency_width: float=None,
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

def apply_gaintable(vis: BlockVisibility, gt: GainTable, inverse=False, **kwargs):
    """Apply a gain table to a block visibility
    
    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis

    if vis.polarisation_frame.type == Polarisation_Frame('stokesI').type:
        for chunk, rows in enumerate(vis_timeslice_iter(vis)):
            visslice = create_blockvisibility_from_rows(vis, rows)
            vistime = numpy.average(visslice.time)
            integration_time = numpy.average(visslice.integration_time)
            gaintable_rows = abs(gt.time - vistime) < integration_time / 2.0
            
            # Find the gain for this set of visibilities
            gain = gt.data['gain'][gaintable_rows]
            gwt = gt.data['weight'][gaintable_rows]
            if inverse:
                gain[gwt>0.0] = 1.0 / gain[gwt>0.0]
                
            original = visslice.data['vis']
            applied = copy.deepcopy(original)
            for a1 in range(visslice.nants-1):
                for a2 in range(a1+1, visslice.nants):
                    applied[:, a2, a1, :, :] = gain[0, a1,:,:] * numpy.conjugate(gain[0, a2,:,:]) * \
                                               original[:, a2, a1, :, :]
                                                         
            vis.data['vis'][rows] = applied
    return vis


def solve_gaintable(vis: BlockVisibility, modelvis: BlockVisibility,**kwargs):
    """Solve a gain table to a block visibility

    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis
    
    gt = create_gaintable_from_blockvisibility(vis)
    
    if vis.polarisation_frame.type == Polarisation_Frame('stokesI').type:
        for chunk, rows in enumerate(vis_timeslice_iter(vis)):
            visslice = create_blockvisibility_from_rows(vis, rows)
            mvisslice = create_blockvisibility_from_rows(modelvis, rows)

            # Form the point source equivalent visibility
            X = numpy.zeros_like(visslice.data['vis'])
            Xwt = numpy.abs(mvisslice.data['vis'])**2 * mvisslice.data['weight']
            mask = Xwt > 0.0
            X[mask]= visslice.data['vis'][mask]/mvisslice.data['vis'][mask]
                        
            # Now average over time, chan. The axes of X are time, antenna2, antenna1, chan, pol
            
            Xave = numpy.average(X * Xwt, axis=(0))
            XwtAve = numpy.average(Xwt, axis=(0))
            
            mask = XwtAve>0.0
            Xave[mask] = Xave[mask]/XwtAve[mask]
            
            gt.data['gain'][chunk,...], gt.data['weight'][chunk,...] = solve_station_gains(Xave, XwtAve)

    return gt

def solve_station_gains(X, Xwt, niter=10, tol=1e-12, phase_only=True, refant=0):
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
        for ant2 in range(ant1+1,nants):
            X[ant1, ant2, ...] = numpy.conjugate(X[ant2, ant1, ...])
            Xwt[ant1, ant2, ...] = Xwt[ant2, ant1, ...]

    def gain_substitution(gain, v, wt):

        nants = gain.shape[0]
        g = copy.deepcopy(gain)
        gwt = numpy.zeros_like(g, dtype='float')

        g2 = (g * numpy.conjugate(g)).real
        for ant1 in range(nants):
            top = 0.0
            bot = 0.0
            for ant2 in range(nants):
                top += g[ant2,...] * v[ant2,ant1,...] * wt[ant2,ant1,...]
                bot += g2[ant2,...] * wt[ant2,ant1,...]
            gain[ant1,...] = top / bot
            gwt[ant1,...] = bot
        return gain, gwt
    
    gainshape = X.shape[1:]
    gain = numpy.ones(shape=gainshape, dtype=X.dtype)
    for iter in range(niter):
        gainLast = gain
        gain, gwt = gain_substitution(gain, X, Xwt)
        if phase_only:
            gain = gain / numpy.abs(gain)
        # gref = numpy.conjugate(gain[0,...]/numpy.abs(gain[0,...]))
        # gain *= gref
        # gain += 0.5 * gainLast
        change = numpy.max(numpy.abs(gain-gainLast))
        if change < tol:
            return gain, gwt
    
    return gain, gwt
