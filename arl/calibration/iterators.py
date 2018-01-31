""" GainTable iterators for iterating through a GainTable

A typical use would be to make a sequence of snapshot images::

    for rows in gaintable_timeslice_iter(vt):
        visslice = create_GainTable_from_rows(vt, rows)
        dirtySnapshot = create_image_from_GainTable(visslice, npixel=512, cellsize=0.001, npol=1)
        dirtySnapshot, sumwt = invert_2d(visslice, dirtySnapshot)


"""

import logging
from typing import Union

import numpy

from arl.data.data_models import GainTable
from arl.data.parameters import get_parameter

log = logging.getLogger(__name__)

def gaintable_null_iter(gt: GainTable, **kwargs) -> numpy.ndarray:
    """One time iterator returning true for all rows
    
    :param gt:
    :param kwargs:
    :return:
    """
    yield numpy.ones_like(gt.time, dtype=bool)


def gaintable_timeslice_iter(gt: GainTable, **kwargs) -> numpy.ndarray:
    """ W slice iterator

    :param wstack: wstack (wavelengths)
    :param gt_slices: Number of slices (second in precedence to wstack)
    :return: Boolean array with selected rows=True
    """
    assert isinstance(gt, GainTable)
    timemin = numpy.min(gt.time)
    timemax = numpy.max(gt.time)
    
    timeslice = get_parameter(kwargs, "timeslice", 'auto')
    if timeslice == 'auto':
        boxes = numpy.unique(gt.time)
        timeslice = 0.1
    elif timeslice is None:
        timeslice = timemax - timemin
        boxes = [0.5*(timemax+timemin)]
    elif isinstance(timeslice, float) or isinstance(timeslice, int):
        boxes = numpy.arange(timemin, timemax, timeslice)
    else:
        gt_slices = get_parameter(kwargs, "gaintable_slices", None)
        assert gt_slices is not None, "Time slicing not specified: set either timeslice or gt_slices"
        boxes = numpy.linspace(timemin, timemax, gt_slices)
        if gt_slices > 1:
            timeslice = boxes[1] - boxes[0]
        else:
            timeslice = timemax - timemin

    for box in boxes:
        rows = numpy.abs(gt.time - box) <= 0.5 * timeslice
        yield rows