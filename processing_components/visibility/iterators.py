""" Visibility iterators for iterating through a BlockVisibility or Visibility.

A typical use would be to make a sequence of snapshot images::

    for rows in vis_timeslice_iter(vt):
        visslice = create_visibility_from_rows(vt, rows)
        dirtySnapshot = create_image_from_visibility(visslice, npixel=512, cellsize=0.001, npol=1)
        dirtySnapshot, sumwt = invert_2d(visslice, dirtySnapshot)

"""

import logging
from typing import Union


import numpy

from data_models.memory_data_models import Visibility, BlockVisibility

log = logging.getLogger(__name__)


def vis_null_iter(vis: Union[Visibility, BlockVisibility], vis_slices=1) -> numpy.ndarray:
    """One time iterator returning true for all rows
    
    :param vis:
    :param vis_slices:
    :return:
    """
    assert vis is not None
    assert isinstance(vis, Visibility) or isinstance(vis, BlockVisibility), vis
    yield numpy.ones_like(vis.time, dtype=bool)


def vis_timeslice_iter(vis: Union[Visibility, BlockVisibility], vis_slices=None) -> numpy.ndarray:
    """ Time slice iterator

    :param vis:
    :param vis_slices: Number of time slices
    :return: Boolean array with selected rows=True
    """
    assert vis is not None
    assert isinstance(vis, Visibility) or isinstance(vis, BlockVisibility), vis
    timemin = numpy.min(vis.time)
    timemax = numpy.max(vis.time)
    
    if vis_slices is None:
        vis_slices = vis_timeslices(vis, 'auto')
    
    boxes = numpy.linspace(timemin, timemax, vis_slices)
    if vis_slices > 1:
        timeslice = boxes[1] - boxes[0]
    else:
        timeslice = timemax - timemin
    
    for box in boxes:
        rows = numpy.abs(vis.time - box) <= 0.5 * timeslice
        yield rows


def vis_timeslices(vis: Visibility, timeslice='auto') -> int:
    """ Calculate number of time slices in a visibility

    :param vis: Visibility
    :param timeslice: 'auto' or float (seconds)
    :return: Number of slices
    """
    assert isinstance(vis, Visibility) or isinstance(vis, BlockVisibility), vis

    timemin = numpy.min(vis.time)
    timemax = numpy.max(vis.time)
    
    if timeslice == 'auto':
        return len(numpy.unique(vis.time))
    else:
        return numpy.ceil(timemax - timemin) / timeslice


def vis_wslices(vis: Visibility, wslice=10.0) -> int:
    """ Calculate number of w slices (or stack) in a visibility

    :param vis: Visibility
    :param wslice: width of w slice (in lambda)
    :return: Number of slices
    """
    assert isinstance(vis, Visibility), vis
    wmaxabs = numpy.max(numpy.abs(vis.w))
    
    return 1 + 2 * numpy.round(wmaxabs / wslice).astype('int')

def vis_wslice_iter(vis: Visibility, vis_slices=1) -> numpy.ndarray:
    """ W slice iterator

    :param vis:
    :param vis_slices: Number of slices
    :return: Boolean array with selected rows=True
    """
    assert isinstance(vis, Visibility), vis
    wmaxabs = numpy.max(numpy.abs(vis.w))
    
    boxes = numpy.linspace(- wmaxabs, +wmaxabs, vis_slices)
    if vis_slices > 1:
        wstack = boxes[1] - boxes[0]
    else:
        wstack = 2 * wmaxabs
    
    for box in boxes:
        rows = numpy.abs(vis.w - box) < 0.5 * wstack
        yield rows