""" Functions for calibration, including creation of gaintables, application of gaintables, and
merging gaintables.

"""

import copy

import numpy.linalg

from arl.data.data_models import GainTable, BlockVisibility, QA, assert_vis_gt_compatible
from arl.data.data_models import ReceptorFrame
from arl.visibility.iterators import vis_timeslice_iter

import logging

log = logging.getLogger(__name__)


def gaintable_summary(gt: GainTable):
    """Return string summarizing the Gaintable

    """
    return "%s rows, %.3f GB" % (gt.data.shape, gt.size())


def create_gaintable_from_blockvisibility(vis: BlockVisibility, timeslice = None,
                                          frequencyslice: float = None, **kwargs) -> GainTable:
    """ Create gain table from visibility.
    
    This makes an empty gain table consistent with the BlockVisibility.
    
    :param vis: BlockVisibilty
    :param timeslice: Time interval between solutions (s)
    :param frequency_width: Frequency solution width (Hz)
    :return: GainTable
    
    """
    assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis
    
    nants = vis.nants
    
    if timeslice is None or timeslice == 'auto':
        utimes = numpy.unique(vis.time)
        ntimes = len(utimes)
        gain_interval = numpy.zeros([ntimes])
        if ntimes > 1:
            gain_interval[:-1] =utimes[1:]- utimes[0:-1]
            gain_interval[-1] =utimes[-1]- utimes[-2]
        else:
            gain_interval[...] = 1.0
    
    else:
        ntimes = numpy.ceil((numpy.max(vis.time) - numpy.min(vis.time))/timeslice).astype('int')
        utimes = numpy.linspace(numpy.min(vis.time), numpy.max(vis.time), ntimes)
        gain_interval = timeslice * numpy.ones([ntimes])
    
    log.debug('create_gaintable_from_blockvisibility: times are %s' % str(utimes))
    log.debug('create_gaintable_from_blockvisibility: intervals are %s' % str(gain_interval))

    ntimes = len(utimes)
    ufrequency = numpy.unique(vis.frequency)
    nfrequency = len(ufrequency)
    
    receptor_frame = ReceptorFrame(vis.polarisation_frame.type)
    nrec = receptor_frame.nrec
    
    gainshape = [ntimes, nants, nfrequency, nrec, nrec]
    gain = numpy.ones(gainshape, dtype='complex')
    if nrec > 1:
        gain[..., 0, 1] = 0.0
        gain[..., 1, 0] = 0.0
    
    gain_weight = numpy.ones(gainshape)
    gain_time = utimes
    gain_frequency = ufrequency
    gain_residual = numpy.zeros([ntimes, nfrequency, nrec, nrec])
    
    gt = GainTable(gain=gain, time=gain_time, interval=gain_interval, weight=gain_weight, residual=gain_residual,
                   frequency=gain_frequency,
                   receptor_frame=receptor_frame)
    
    assert isinstance(gt, GainTable), "gt is not a GainTable: %r" % gt
    assert_vis_gt_compatible(vis, gt)

    return gt


def apply_gaintable(vis: BlockVisibility, gt: GainTable, inverse=False, **kwargs) -> BlockVisibility:
    """Apply a gain table to a block visibility
    
    The corrected visibility is::
    
        V_corrected = {g_i * g_j^*}^-1 V_obs
        
    If the visibility data are polarised e.g. polarisation_frame("linear") then the inverse operator
    represents an actual inverse of the gains.
    
    :param vis: Visibility to have gains applied
    :param gt: Gaintable to be applied
    :param inverse: Apply the inverse (default=False)
    :return: input vis with gains applied
    
    """
    assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis
    assert isinstance(gt, GainTable), "gt is not a GainTable: %r" % gt

    assert_vis_gt_compatible(vis, gt)
    
    if inverse:
        log.debug('apply_gaintable: Apply inverse gaintable')
    else:
        log.debug('apply_gaintable: Apply gaintable')
    
    is_scalar = gt.gain.shape[-2:] == (1, 1)
    if is_scalar:
        log.debug('apply_gaintable: scalar gains')
        
    for chunk, rows in enumerate(vis_timeslice_iter(vis, **kwargs)):
        if numpy.sum(rows) > 0:
            vistime = numpy.average(vis.time[rows])
            gaintable_rows = abs(gt.time - vistime) < gt.interval / 2.0
            
            # Lookup the gain for this set of visibilities
            gain = gt.data['gain'][gaintable_rows]
            
            # The shape of the mueller matrix is
            ntimes, nant, nchan, nrec, _ = gain.shape
            
            original = vis.vis[rows]
            applied = copy.deepcopy(original)
            for time in range(ntimes):
                for a1 in range(vis.nants - 1):
                    for a2 in range(a1 + 1, vis.nants):
                        if is_scalar:
                            smueller = gain[time, a1, :, 0] * numpy.conjugate(gain[time, a2, :, 0])
                            if inverse:
                                if numpy.abs(smueller).all() > 0.0:
                                    applied[time, a2, a1, :, 0][..., numpy.newaxis] = original[time, a2, a1, :, 0][..., numpy.newaxis] / smueller
                            else:
                                applied[time, a2, a1, :, 0][..., numpy.newaxis] = original[time, a2, a1, :, 0][..., numpy.newaxis] * smueller
                        else:
                            for chan in range(nchan):
                                mueller = numpy.kron(gain[time, a1, chan, :, :], numpy.conjugate(gain[time, a2, chan, :, :]))
                                if inverse:
                                    # If the Mueller is singular, ignore it
                                    try:
                                        mueller = numpy.linalg.inv(mueller)
                                        applied[time, a2, a1, chan, :] = numpy.matmul(mueller, original[time, a2, a1, chan, :])
                                    except numpy.linalg.linalg.LinAlgError:
                                        applied[time, a2, a1, chan, :] = original[time, a2, a1, chan, :]
                                else:
                                    applied[time, a2, a1, chan, :] = numpy.matmul(mueller, original[time, a2, a1, chan, :])
            
            vis.data['vis'][rows] = applied
    return vis


def append_gaintable(gt: GainTable, othergt: GainTable) -> GainTable:
    """Append othergt to gt

    :param gt:
    :param othergt:
    :return: GainTable gt + othergt
    """
    assert gt.receptor_frame == othergt.receptor_frame
    gt.data = numpy.hstack((gt.data, othergt.data))
    return gt


def copy_gaintable(gt: GainTable, zero=False) -> GainTable:
    """Copy a GainTable

    Performs a deepcopy of the data array
    """
    assert isinstance(gt, GainTable), gt
    
    newgt = copy.copy(gt)
    newgt.data = copy.deepcopy(gt.data)
    if zero:
        newgt.data['gt'][...] = 0.0
    return newgt


def create_gaintable_from_rows(gt: GainTable, rows: numpy.ndarray, makecopy=True) -> GainTable:
    """ Create a GainTable from selected rows

    :param gt: GainTable
    :param rows: Boolean array of row selection
    :param makecopy: Make a deep copy (True)
    :return: GainTable
    """
    
    if rows is None or numpy.sum(rows) == 0:
        return None
    
    assert len(rows) == gt.ntimes, "Length of rows does not agree with length of GainTable"
    
    assert isinstance(gt, GainTable), gt
        
    if makecopy:
        newgt = copy_gaintable(gt)
        newgt.data = copy.deepcopy(gt.data[rows])
        return newgt
    else:
        gt.data = copy.deepcopy(gt.data[rows])
        
        return gt


def qa_gaintable(gt: GainTable, context=None) -> QA:
    """Assess the quality of a gaintable

    :param gt:
    :return: AQ
    """
    agt = numpy.abs(gt.gain)
    pgt = numpy.angle(gt.gain)
    data = {'shape': gt.gain.shape,
            'maxabs-amp': numpy.max(agt),
            'minabs-amp': numpy.min(agt),
            'rms-amp': numpy.std(agt),
            'medianabs-amp': numpy.median(agt),
            'maxabs-phase': numpy.max(pgt),
            'minabs-phase': numpy.min(pgt),
            'rms-phase': numpy.std(pgt),
            'medianabs-phase': numpy.median(pgt),
            'residual': numpy.max(gt.residual)
            }
    return QA(origin='qa_gaintable', data=data, context=context)
