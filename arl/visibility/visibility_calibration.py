# Tim Cornwell <realtimcornwell@gmail.com>
#

#

import logging

log = logging.getLogger("arl.visibility_calibration")

from data.data_models import *
from data.parameters import *

"""
Functions that either solve_gains for the calibration or apply it. On solution the gains are written into a gaintable. For
correction, the gaintable is read and, if necessary, interpolated.
"""


def create_gaintable_from_array(gain: numpy.array, time: numpy.array, antenna: numpy.array, weight: numpy.array,
                                frequency: numpy.array, copy=False, meta=None, params=None):
    """ Create a gaintable from arrays

    :param gain:
    :param time:
    :param antenna:
    :param weight:
    :param frequency:
    :param copy:
    :param meta:
    :param params: Dictionary containing parameters
    :returns: Gaintable
    """
    if params is None:
        params = {}
    log_parameters(params)
    if meta is None:
        meta = {}
    nrows = time.shape[0]
    assert len(frequency) == gain.shape[1], "Discrepancy in frequency channels"
    assert len(antenna) == nrows, "Discrepancy in number of antenna rows"
    assert gain.shape[0] == nrows, "Discrepancy in number of gain rows"
    assert weight.shape[0] == nrows, "Discrepancy in number of weight rows"
    fg = GainTable()
    
    fg.data = Table(data=[gain, time, antenna, weight], names=['gain', 'time', 'antenna', 'weight'], copy=copy,
                    meta=meta)
    fg.frequency = frequency
    return fg


def interpolate_gaintable(gt: GainTable, params=None):
    """ Interpolate a GainTable to new sampling

    :param gt: GainTable
    :param params: Dictionary containing parameters
    :returns: Gaintable
    """
    if params is None:
        params = {}
    log_parameters(params)
    # TODO: implement
    
    log.error('"interpolate_gaintable: not yet implemented')
    return GainTable()


def solve_gains(vis: Visibility, sm: Skymodel, params=None) -> GainTable:
    """ Solve for calibration using a sky model
    
    :param params:
    :param vis:
    :param sm:
    :returns: GainTable
    """
    if params is None:
        params = {}
    log_parameters(params)
    # TODO: Implement calibration solution
    log.error("solve_gains: not yet implemented")
    return GainTable()


def correct_visibility(vis: Visibility, gt: GainTable, params=None) -> Visibility:
    """ Correct a vistable using a GainTable

    :param params:
    :param vis: Visibility to be processed
    :param gt: GainTable
    :returns: Visibility
    """
    # TODO: Implement calibration application
    if params is None:
        params = {}
    log_parameters(params)
    log.error("correct_visibility: not yet implemented")
    return vis


def peel_skycomponent(vis: Visibility, sc: Skycomponent, params=None) -> Visibility:
    """ Correct a vistable using a GainTable

    :param params:
    :param vis: Visibility to be processed
    :param sc:
    :returns: Visibility, GainTable
    """
    # TODO: Implement peeling
    if params is None:
        params = {}
    log_parameters(params)
    log.error("peel_skycomponent: not yet implemented")
    return vis


def qa_gaintable(gt, params=None):
    """Assess the quality of a gaintable

    :param gt:
    :param params:
    :returns: AQ
    """
    # TODO: implement
    
    if params is None:
        params = {}
    log_parameters(params)
    log.error("qa_gaintable: not yet implemented")
    return QA()
