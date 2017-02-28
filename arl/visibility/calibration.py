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
