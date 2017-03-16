""" Pipeline functionss

"""
# Tim Cornwell <realtimcornwell@gmail.com>
#
#
import collections

from arl.calibration.solvers import solve_gaintable
from arl.fourier_transforms.ftprocessor import predict_skycomponent_blockvisibility
from arl.visibility.operations import *

log = logging.getLogger(__name__)


def rcal(vis: BlockVisibility, components, **kwargs):
    """ Real-time calibration pipeline.
    
    :param vis: Visibility or Union(Visibility, Iterable)
    :param comp: Component-based sky model
    :return: gaintable
   """
    
    if not isinstance(vis, collections.Iterable):
        vis = [vis]
    
    for ichunk, vischunk in enumerate(vis):
        vispred = copy_visibility(vischunk)
        vispred.data['vis'][...] = 0.0
        vispred = predict_skycomponent_blockvisibility(vispred, components)
        gt = solve_gaintable(vischunk, vispred, phase_only=False)
        yield gt


def ical(**kwargs):
    """ Post observation image, deconvolve, and self-calibrate
   
    :param params: Dictionary containing parameters
    :return:
    """
    # TODO: implement
    
    return True


def continuum_imaging(**kwargs):
    """Continuum imaging from calibrated (DDE and DIE) data

    
    :param params: Dictionary containing parameters
    :return:
    """
    
    # TODO: implement
    
    return True


def spectral_line_imaging(**kwargs):
    """Spectral line imaging from calibrated (DDE and DIE) data
    
    :param params: Dictionary containing parameters
    :return:
    """
    # TODO: implement
    
    return True


def fast_imaging(**kwargs):
    """Fast imaging from calibrated (DIE only) data

    :param params: Dictionary containing parameters
    :return:
    """
    # TODO: implement
    
    return True


def eor(**kwargs):
    """eor calibration and imaging
    
    :param params: Dictionary containing parameters
    :return:
    """
    # TODO: implement
    
    return True
