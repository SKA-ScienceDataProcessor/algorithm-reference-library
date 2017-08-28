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

from arl.data.data_models import Visibility, BlockVisibility
from arl.data.parameters import get_parameter

log = logging.getLogger(__name__)


def vis_timeslice_iter(vis: Visibility, **kwargs) -> numpy.ndarray:
    """ W slice iterator

    :param wstack: wstack (wavelengths)
    :param vis_slices: Number of slices (second in precedence to wstack)
    :return: Boolean array with selected rows=True
    """
    assert type(vis) == Visibility or type(vis) == BlockVisibility
    timemin = numpy.min(vis.time)
    timemax = numpy.max(vis.time)
    
    timeslice = get_parameter(kwargs, "timeslice", None)
    if timeslice is None or timeslice == 'auto':
        vis_slices = get_parameter(kwargs, "vis_slices", None)
        if vis_slices is None:
            vis_slices = len(numpy.unique(vis.time))
        boxes = numpy.linspace(timemin, timemax, vis_slices)
        timeslice = (timemax - timemin) / vis_slices
    else:
        vis_slices = 1 + 2 * numpy.round((timemax - timemin) / timeslice).astype('int')
        boxes = numpy.linspace(timemin, timemax, vis_slices)
    
    for box in boxes:
        rows = numpy.abs(vis.time - box) <= 0.5 * timeslice
        yield rows


def vis_wstack_iter(vis: Visibility, **kwargs) -> numpy.ndarray:
    """ W slice iterator

    :param wstack: wstack (wavelengths)
    :param vis_slices: Number of slices (second in precedence to wstack)
    :return: Boolean array with selected rows=True
    """
    assert type(vis) == Visibility or type(vis) == BlockVisibility
    wmaxabs = (numpy.max(numpy.abs(vis.w)))
    
    wstack = get_parameter(kwargs, "wstack", None)
    if wstack is None:
        vis_slices = get_parameter(kwargs, "vis_slices", 1)
        boxes = numpy.linspace(- wmaxabs, +wmaxabs, vis_slices)
        wstack = 2 * wmaxabs / vis_slices
    else:
        vis_slices = 1 + 2 * numpy.round(wmaxabs / wstack).astype('int')
        boxes = numpy.linspace(- wmaxabs, +wmaxabs, vis_slices)
    
    for box in boxes:
        rows = numpy.abs(vis.w - box) < 0.5 * wstack
        yield rows


def vis_slice_iter(vis: Union[Visibility, BlockVisibility], **kwargs) -> numpy.ndarray:
    """ Iterates in slices

    :param step: Size of step to be iterated over (in rows)
    :param vis_slices: Number of slices (second in precedence to step)
    :return: Boolean array with selected rows=True

    """
    assert type(vis) == Visibility or type(vis) == BlockVisibility
    
    step = get_parameter(kwargs, "step", None)
    if step is None:
        vis_slices = get_parameter(kwargs, "vis_slices", 1)
        step = 1 + vis.nvis // vis_slices
    
    assert step > 0
    for row in range(0, vis.nvis, step):
        yield range(row, min(row + step, vis.nvis))
