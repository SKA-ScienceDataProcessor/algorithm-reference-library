""" Visibility iterators for iterating through a BlockVisibility or Visibility.

A typical use would be to make a sequence of snapshot images::

    for rows in vis_timeslice_iter(vt):
        visslice = create_visibility_from_rows(vt, rows)
        dirtySnapshot = create_image_from_visibility(visslice, npixel=512, cellsize=0.001, npol=1)
        dirtySnapshot, sumwt = invert_2d(visslice, dirtySnapshot)


"""

import logging

import numpy

from arl.data.parameters import get_parameter
from arl.data.data_models import *

log = logging.getLogger(__name__)


def vis_timeslice_iter(vis, **kwargs):
    """ Time slice iterator
    
    If timeslice='auto' then timeslice is taken to be the difference between the first two
    unique elements of the vis time.
          
    :param timeslice: Timeslice (seconds) ('auto')
    :returns: Boolean array with selected rows=True
        
    """
    
    assert type(vis) == Visibility or type(vis) == BlockVisibility
    
    uniquetimes = numpy.unique(vis.time)
    timeslice = get_parameter(kwargs, "timeslice", 'auto')
    if timeslice == 'auto':
        log.debug('vis_timeslice_iter: Found %d unique times' % len(uniquetimes))
        if len(uniquetimes) > 1:
            timeslice = (uniquetimes[1] - uniquetimes[0])
            log.debug('vis_timeslice_auto: Guessing time interval to be %.2f s' % timeslice)
        else:
            # Doesn't matter what we set it to.
            timeslice = vis.integration_time[0]
    boxes = timeslice * numpy.round(uniquetimes / timeslice).astype('int')
        
    for box in boxes:
        rows = numpy.abs(vis.time - box) < 0.5 * timeslice
        yield rows


def vis_wslice_iter(vis, **kwargs):
    """ W slice iterator

    :param wslice: wslice (wavelengths) (must be specified)
    :returns: Boolean array with selected rows=True
    """
    assert type(vis) == Visibility or type(vis) == BlockVisibility

    wslice = get_parameter(kwargs, "wslice", None)
    assert wslice is not None, "wslice must be specified"

    wmaxabs = (numpy.max(numpy.abs(vis.w)))
    nboxes = 1 + 2 * numpy.round(wmaxabs / wslice).astype('int')
    boxes = numpy.linspace(- wmaxabs, +wmaxabs, nboxes)
    
    for box in boxes:
        rows = numpy.abs(vis.w - box) < 0.5 * wslice
        if numpy.sum(rows) > 0:
            yield rows


def vis_slice_iter(vis, **kwargs):
    """ Iterates in slices

    :param step: Size of step to be iterated over (in rows)
    :returns: Boolean array with selected rows=True

    """
    assert type(vis) == Visibility or type(vis) == BlockVisibility

    step = get_parameter(kwargs, "step", None)
    assert step is not None, "step must be specified"
    assert step > 0
    for row in range(0, vis.nvis, step):
            yield range(row, min(row+step, vis.nvis))


