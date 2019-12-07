""" Functions for calibration, including creation of pointingtables, application of pointingtables, and
merging pointingtables.

"""

__all__ = ['create_pointingtable_from_rows', 'create_pointingtable_from_blockvisibility', 'copy_pointingtable']

import copy
import logging

import numpy.linalg

from data_models.memory_data_models import PointingTable, BlockVisibility, QA
from data_models.memory_data_models import ReceptorFrame

from processing_library.util.coordinate_support import hadec_to_azel

log = logging.getLogger(__name__)

def pointingtable_summary(pt: PointingTable):
    """Return string summarizing the Gaintable

    """
    return "%s rows, %.3f GB" % (pt.data.shape, pt.size())


def create_pointingtable_from_blockvisibility(vis: BlockVisibility, pointing_frame='azel', timeslice=None,
                                              frequencyslice: float = None, **kwargs) -> PointingTable:
    """ Create pointing table from visibility.
    
    This makes an empty pointing table consistent with the BlockVisibility.
    
    :param vis: BlockVisibilty
    :param timeslice: Time interval between solutions (s)
    :param frequency_width: Frequency solution width (Hz)
    :return: PointingTable
    
    """
    assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis
    
    nants = vis.nants
    
    if timeslice is None or timeslice == 'auto':
        utimes = numpy.unique(vis.time)
        ntimes = len(utimes)
        pointing_interval = numpy.zeros([ntimes])
        if ntimes > 1:
            pointing_interval[:-1] = utimes[1:] - utimes[0:-1]
            pointing_interval[-1] = utimes[-1] - utimes[-2]
        else:
            pointing_interval[...] = 1.0
    
    else:
        ntimes = numpy.ceil((numpy.max(vis.time) - numpy.min(vis.time)) / timeslice).astype('int')
        utimes = numpy.linspace(numpy.min(vis.time), numpy.max(vis.time), ntimes)
        pointing_interval = timeslice * numpy.ones([ntimes])
    
    #    log.debug('create_pointingtable_from_blockvisibility: times are %s' % str(utimes))
    #    log.debug('create_pointingtable_from_blockvisibility: intervals are %s' % str(pointing_interval))
    
    ntimes = len(utimes)
    ufrequency = numpy.unique(vis.frequency)
    nfrequency = len(ufrequency)
    
    receptor_frame = ReceptorFrame(vis.polarisation_frame.type)
    nrec = receptor_frame.nrec
    
    pointingshape = [ntimes, nants, nfrequency, nrec, 2]
    pointing = numpy.zeros(pointingshape)
    if nrec > 1:
        pointing[..., 0, 0] = 0.0
        pointing[..., 1, 0] = 0.0
        pointing[..., 0, 1] = 0.0
        pointing[..., 1, 1] = 0.0

    ha = numpy.pi * vis.time / 43200.0
    dec = vis.phasecentre.dec.rad
    latitude = vis.configuration.location.lat.rad
    az, el = hadec_to_azel(ha, dec, latitude)

    pointing_nominal = numpy.zeros([ntimes, nants, nfrequency, nrec, 2])
    pointing_nominal[...,0] = az[:, numpy.newaxis, numpy.newaxis, numpy.newaxis]
    pointing_nominal[...,1] = el[:, numpy.newaxis, numpy.newaxis, numpy.newaxis]
    pointing_weight = numpy.ones(pointingshape)
    pointing_time = utimes
    pointing_frequency = ufrequency
    pointing_residual = numpy.zeros([ntimes, nfrequency, nrec, 2])
    pointing_frame = pointing_frame
    
    pt = PointingTable(pointing=pointing, nominal=pointing_nominal,
                       time=pointing_time, interval=pointing_interval, weight=pointing_weight,
                       residual=pointing_residual, frequency=pointing_frequency, receptor_frame=receptor_frame,
                       pointing_frame=pointing_frame, pointingcentre=vis.phasecentre, configuration=vis.configuration)
    
    assert isinstance(pt, PointingTable), "pt is not a PointingTable: %r" % pt
    
    return pt


def copy_pointingtable(pt: PointingTable, zero=False):
    """Copy a PointingTable

    Performs a deepcopy of the data array
    """
    
    if pt is None:
        return pt
    
    assert isinstance(pt, PointingTable), pt
    
    newpt = copy.copy(pt)
    newpt.data = copy.deepcopy(pt.data)
    if zero:
        newpt.data['pt'][...] = 0.0
    return newpt


def create_pointingtable_from_rows(pt: PointingTable, rows: numpy.ndarray, makecopy=True) -> PointingTable:
    """ Create a PointingTable from selected rows

    :param pt: PointingTable
    :param rows: Boolean array of row selection
    :param makecopy: Make a deep copy (True)
    :return: PointingTable
    """
    
    if rows is None or numpy.sum(rows) == 0:
        return None
    
    assert len(rows) == pt.ntimes, "Lenpth of rows does not agree with lenpth of PointingTable"
    
    assert isinstance(pt, PointingTable), pt
    
    if makecopy:
        newpt = copy_pointingtable(pt)
        newpt.data = copy.deepcopy(pt.data[rows])
        return newpt
    else:
        pt.data = copy.deepcopy(pt.data[rows])
        
        return pt


def qa_pointingtable(pt: PointingTable, context=None) -> QA:
    """Assess the quality of a pointingtable

    :param pt:
    :return: AQ
    """
    apt = numpy.abs(pt.pointing[pt.weight > 0.0])
    ppt = numpy.angle(pt.pointing[pt.weight > 0.0])
    data = {'shape': pt.pointing.shape,
            'maxabs-amp': numpy.max(apt),
            'minabs-amp': numpy.min(apt),
            'rms-amp': numpy.std(apt),
            'medianabs-amp': numpy.median(apt),
            'maxabs-phase': numpy.max(ppt),
            'minabs-phase': numpy.min(ppt),
            'rms-phase': numpy.std(ppt),
            'medianabs-phase': numpy.median(ppt),
            'residual': numpy.max(pt.residual)
            }
    return QA(origin='qa_pointingtable', data=data, context=context)
