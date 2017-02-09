# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that either solve_gains for the calibration or apply
it. On solution the gains are written into a gaintable. For
correction, the gaintable is read and, if necessary, interpolated.
"""

import logging

from arl.data.data_models import *
from arl.data.parameters import *
log = logging.getLogger("visibility.calibration")

def interpolate_gaintable(gt: GainTable, **kwargs):
    """ Interpolate a GainTable to new sampling

    :param gt: GainTable
    :param params: Dictionary containing parameters
    :returns: Gaintable
    """
    # TODO: implement
    
    raise RuntimeError('"interpolate_gaintable: not yet implemented')
    return GainTable()


def solve_gains(vis: Visibility, sm: Skymodel, **kwargs) -> GainTable:
    """ Solve for calibration using a sky model
    
    :param params:
    :param vis:
    :param sm:
    :returns: GainTable
    """
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis

    # TODO: Implement calibration solution
    raise RuntimeError("solve_gains: not yet implemented")
    return GainTable()


def correct_visibility(vis: Visibility, gt: GainTable, **kwargs) -> Visibility:
    """ Correct a vistable using a GainTable

    :param params:
    :param vis: Visibility to be processed
    :param gt: GainTable
    :returns: Visibility
    """
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis

    # TODO: Implement calibration application
    raise RuntimeError("correct_visibility: not yet implemented")
    return vis


def peel_skycomponent(vis: Visibility, sc: Skycomponent, **kwargs) -> Visibility:
    """ Correct a vistable using a GainTable

    :param params:
    :param vis: Visibility to be processed
    :param sc:
    :returns: Visibility, GainTable
    """
    # TODO: Implement peeling
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis

    raise RuntimeError("peel_skycomponent: not yet implemented")
    return vis


def qa_gaintable(gt, **kwargs):
    """Assess the quality of a gaintable

    :param gt:
    :param params:
    :returns: AQ
    """
    # TODO: implement
    raise RuntimeError("qa_gaintable: not yet implemented")
    return QA()
