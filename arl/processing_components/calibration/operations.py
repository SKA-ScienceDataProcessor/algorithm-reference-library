""" Functions for calibration, including creation of gaintables, application of gaintables, and
merging gaintables.

"""

__all__ = ['gaintable_summary', 'gaintable_plot', 'qa_gaintable', 'apply_gaintable', 'append_gaintable',
           'create_gaintable_from_blockvisibility', 'create_gaintable_from_blockvisibility',
           'create_gaintable_from_rows', 'copy_gaintable']

import copy
import logging

import numpy.linalg

from arl.data_models import GainTable, BlockVisibility, QA, assert_vis_gt_compatible
from arl.data_models import ReceptorFrame
from arl.processing_components.visibility.iterators import vis_timeslice_iter

log = logging.getLogger(__name__)


def gaintable_summary(gt: GainTable):
    """Return string summarizing the Gaintable

    """
    return "%s rows, %.3f GB" % (gt.data.shape, gt.size())


def create_gaintable_from_blockvisibility(vis: BlockVisibility, timeslice=None,
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
        timeslice = numpy.min(vis.integration_time)
    
    utimes = timeslice * numpy.unique(numpy.round((vis.time - vis.time[0]) / timeslice))
    ntimes = len(utimes)
    gain_interval = timeslice * numpy.ones([ntimes])

    #    log.debug('create_gaintable_from_blockvisibility: times are %s' % str(utimes))
    #    log.debug('create_gaintable_from_blockvisibility: intervals are %s' % str(gain_interval))
    
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
    gain_time = utimes + vis.time[0]
    gain_frequency = ufrequency
    gain_residual = numpy.zeros([ntimes, nfrequency, nrec, nrec])
    
    gt = GainTable(gain=gain, time=gain_time, interval=gain_interval, weight=gain_weight, residual=gain_residual,
                   frequency=gain_frequency, receptor_frame=receptor_frame, phasecentre=vis.phasecentre,
                   configuration=vis.configuration)
    
    assert isinstance(gt, GainTable), "gt is not a GainTable: %r" % gt
    assert_vis_gt_compatible(vis, gt)
    
    return gt


def apply_gaintable(vis: BlockVisibility, gt: GainTable, inverse=False, vis_slices=None, **kwargs) -> BlockVisibility:
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
    
    for chunk, rows in enumerate(vis_timeslice_iter(vis, vis_slices=vis_slices)):
        if numpy.sum(rows) > 0:
            vistime = numpy.average(vis.time[rows])
            gaintable_rows = abs(gt.time - vistime) < gt.interval / 2.0
            
            # Lookup the gain for this set of visibilities
            gain = gt.data['gain'][gaintable_rows]
            gainwt = gt.data['weight'][gaintable_rows]
            
            # The shape of the mueller matrix is
            ntimes, nant, nchan, nrec, _ = gain.shape
            
            original = vis.vis[rows]
            originalwt = vis.weight[rows]
            applied = copy.deepcopy(original)
            appliedwt = copy.deepcopy(originalwt)
            for time in range(ntimes):
                antantwt = numpy.outer(gainwt, gainwt)
                if is_scalar:
                    if inverse:
                        lgain = numpy.ones_like(gain)
                        lgain[numpy.abs(gain) > 0.0] = 1.0 / gain[numpy.abs(gain) > 0.0]
                    else:
                        lgain = gain
                    clgain = numpy.conjugate(lgain)
                    for chan in range(nchan):
                        smueller = numpy.ma.outer(lgain[time, :, chan, 0],
                                                  clgain[time, :, chan, 0]).reshape([nant, nant])
                        applied[time, :, :, chan, 0] = original[time, :, :, chan, 0] * smueller
                        antantwt = numpy.outer(gainwt[time, :, chan, 0, 0], gainwt[time, :, chan, 0, 0])
                        applied[time, :, :, chan, 0][antantwt == 0.0] = 0.0
                        appliedwt[time, :, :, chan, 0][antantwt == 0.0] = 0.0
                else:
                    for a1 in range(vis.nants - 1):
                        for a2 in range(a1 + 1, vis.nants):
                            for chan in range(nchan):
                                mueller = numpy.kron(gain[time, a1, chan, :, :],
                                                     numpy.conjugate(gain[time, a2, chan, :, :]))
                                if inverse:
                                    # If the Mueller is singular, ignore it
                                    try:
                                        mueller = numpy.linalg.inv(mueller)
                                        applied[time, a2, a1, chan, :] = \
                                            numpy.matmul(mueller, original[time, a2, a1, chan, :])
                                    except numpy.linalg.linalg.LinAlgError:
                                        applied[time, a2, a1, chan, :] = \
                                            original[time, a2, a1, chan, :]
                                else:
                                    applied[time, a2, a1, chan, :] = \
                                        numpy.matmul(mueller, original[time, a2, a1, chan, :])
                                if (gainwt[time, a1, chan, 0, 0] <= 0.0) or (gainwt[time, a1, chan, 0, 0] <= 0.0):
                                    applied[time, a2, a1, chan, 0] = 0.0
                                    appliedwt[time, a2, a1, chan, 0] = 0.0
            
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


def copy_gaintable(gt: GainTable, zero=False):
    """Copy a GainTable

    Performs a deepcopy of the data array
    """
    
    if gt is None:
        return gt
    
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
    agt = numpy.abs(gt.gain[gt.weight > 0.0])
    pgt = numpy.angle(gt.gain[gt.weight > 0.0])
    rgt = gt.residual[numpy.sum(gt.weight, axis=1) > 0.0]
    data = {'shape': gt.gain.shape,
            'maxabs-amp': numpy.max(agt),
            'minabs-amp': numpy.min(agt),
            'rms-amp': numpy.std(agt),
            'medianabs-amp': numpy.median(agt),
            'maxabs-phase': numpy.max(pgt),
            'minabs-phase': numpy.min(pgt),
            'rms-phase': numpy.std(pgt),
            'medianabs-phase': numpy.median(pgt),
            'residual': numpy.max(rgt)
            }
    return QA(origin='qa_gaintable', data=data, context=context)


def gaintable_plot(gt: GainTable, ax, title='', value='amp', ants=None,  channels=None,
                   label_max=10, **kwargs):
    """ Standard plot of gain table

    :param gt: Gaintable
    :param ax: matplotlib axes
    :param value: 'amp' or 'phase' or 'residual'
    :param ants: Antennas to plot
    :param channels: Channels to plot
    :param kwargs:
    :return:
    """
    if ants is None:
        ants = range(gt.nants)
    if channels is None:
        channels = range(gt.nchan)
    
    if value == "residual":
        residual = gt.residual[:, channels, 0, 0]
        ax.plot(gt.time, residual, '.')
    else:
        for ant in ants:
            if gt.configuration is not None:
                label = gt.configuration.names[ant]
            else:
                label = ''
            amp = numpy.abs(gt.gain[:, ant, channels, 0, 0])
            if value == 'amp':
                ax.plot(gt.time, amp, '.', label=label)
            else:
                angle = numpy.angle(gt.gain[:, ant, channels, 0, 0])
                ax.plot(gt.time, angle, '.', label=label)
   
        if gt.configuration is not None:
            if len(gt.configuration.names) < label_max:
                ax.legend()
                
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(value)
