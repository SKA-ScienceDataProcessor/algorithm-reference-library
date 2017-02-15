# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that either solve_gains for the calibration or apply
it. On solution the gains are written into a gaintable. For
correction, the gaintable is read and, if necessary, interpolated.
"""

from arl.data.data_models import *
from arl.data.parameters import *
log = logging.getLogger("visibility.calibration")


# noinspection PyUnreachableCode
def interpolate_gaintable(gt: GainTable, **kwargs):
    """ Interpolate a GainTable to new sampling

    :param gt: GainTable
    :param params: Dictionary containing parameters
    :returns: Gaintable
    """
    # TODO: implement
    
    raise RuntimeError('"interpolate_gaintable: not yet implemented')
    return GainTable()


# noinspection PyUnreachableCode
def solve_gains(vis: BlockVisibility, sm: Skymodel, **kwargs) -> GainTable:
    """ Solve for calibration using a sky model
    
    :param params:
    :param vis:
    :param sm:
    :returns: GainTable
    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis

    # TODO: Implement calibration solution
    raise RuntimeError("solve_gains: not yet implemented")
    return GainTable()


# noinspection PyUnreachableCode
def correct_blockvisibility(vis: BlockVisibility, gt: GainTable, **kwargs) -> BlockVisibility:
    """ Correct a vistable using a GainTable

    :param params:
    :param vis: BlockVisibility to be processed
    :param gt: GainTable
    :returns: BlockVisibility
    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis

    # TODO: Implement calibration application
    raise RuntimeError("correct_blockvisibility: not yet implemented")
    return vis


# noinspection PyUnreachableCode
def peel_skycomponent(vis: BlockVisibility, sc: Skycomponent, **kwargs) -> BlockVisibility:
    """ Correct a vistable using a GainTable

    :param params:
    :param vis: BlockVisibility to be processed
    :param sc:
    :returns: BlockVisibility, GainTable
    """
    # TODO: Implement peeling
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis

    raise RuntimeError("peel_skycomponent: not yet implemented")
    return vis


# noinspection PyUnreachableCode
def qa_gaintable(gt, **kwargs):
    """Assess the quality of a gaintable

    :param gt:
    :param params:
    :returns: AQ
    """
    # TODO: implement
    raise RuntimeError("qa_gaintable: not yet implemented")
    return QA()
