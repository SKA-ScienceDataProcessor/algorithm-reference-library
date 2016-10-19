# Tim Cornwell <realtimcornwell@gmail.com>
#

import numpy
#
from astropy.coordinates import SkyCoord

from arl.data_models import *
from arl.parameters import *

import logging
log = logging.getLogger("arl.visibility_calibration")

"""
Functions that either solve_gains for the calibration or apply it. On solution the gains are written into a gaintable. For
correction, the gaintable is read and, if necessary, interpolated.
"""


def create_gaintable_from_array(gain: numpy.array, time: numpy.array, antenna: numpy.array, weight: numpy.array,
                                frequency: numpy.array, copy=False, meta=None, params={}):
    """ Create a gaintable from arrays

    :param gain:
    :type GainTable:
    :param time:
    :type numpy.array:
    :param antenna:
    :type numpy.array:
    :param weight:
    :type numpy.array:
    :param frequency:
    :type numpy.array:
    :param copy:
    :type bool:
    :param meta:
    :type dict:
    :param params: Dictionary containing parameters
    :returns: Gaintable
    """
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


def interpolate_gaintable(gt: GainTable, params={}):
    """ Interpolate a GainTable to new sampling

    :param gt: GainTable
    :type GainTable:
    :param params: Dictionary containing parameters
    :returns: Gaintable
    """
    log_parameters(params)
    # TODO: implement
    
    log.error('"interpolate_gaintable: not yet implemented')
    return GainTable()


def solve_gains(vis: Visibility, sm: SkyModel, params={}) -> GainTable:
    """ Solve for calibration using a sky model
    
    :param vis:
    :type Visibility: Visibility to be processed
    :param sm:
    :type SkyModel:
    :returns: GainTable
    """
    log_parameters(params)
    #TODO: Implement calibration solution
    log.error("solve_gains: not yet implemented")
    return GainTable()


def correct_visibility(vis: Visibility, gt: GainTable, params={}) -> Visibility:
    """ Correct a vistable using a GainTable

    :param vis: Visibility to be processed
    :type Visibility:
    :param gt: GainTable
    :type GainTable:
    :returns: Visibility
    """
    # TODO: Implement calibration application
    log_parameters(params)
    log.error("correct_visibility: not yet implemented")
    return vis


def peel_skycomponent(vis: Visibility, sc: SkyComponent, params={}) -> Visibility:
    """ Correct a vistable using a GainTable

    :param vis: Visibility to be processed
    :type Visibility:
    :param sc:
    :type SkyComponent:
    :returns: Visibility, GainTable
    """
    # TODO: Implement peeling
    log_parameters(params)
    log.error("peel_skycomponent: not yet implemented")
    return vis


def qa_gaintable(gt, params={}):
    """Assess the quality of a gaintable

    :param im:
    :type GainTable:
    :returns: AQ
    """
    # TODO: implement

    log_parameters(params)
    log.error("qa_gaintable: not yet implemented")
    return QA()
