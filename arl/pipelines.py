# Tim Cornwell <realtimcornwell@gmail.com>
#
#

from arl.visibility_calibration import solve_gains, qa_gaintable
from arl.quality_assessment import QA
from arl.parameters import get_parameter

def RCAL(params):
    """ Real-time calibration: single shot
     
    :param params:
    :return:
   """
    vis = get_parameter(params, 'visibility', None)
    sm = get_parameter(params, 'skymodel', None)

    gains = solve_gains(vis, sm, params)
    qa = qa_gaintable(gains)
    if qa:
        print("pipelines.RCAL: Solution good")
    return qa


def ICAL(params):
    """ Post observation image, deconvolve, and self-calibrate
   
    :param params:
    :return:
    """
    vis = get_parameter(params, 'visibility')
    sm = get_parameter(params, 'skymodel')

    gains = solve_gains(vis, sm, params)
    qa = qa_gaintable(gains)
    if qa:
        print("pipelines.ICAL: Solution good")
    return qa


def continuum_imaging(params):
    """Continuum imaging from calibrated (DDE and DIE) data

    
    :param params:
    :return:
    """

    vis = get_parameter(params, 'visibility')
    sm = get_parameter(params, 'skymodel')
    
    return QA()
    

def spectral_line_imaging(params):
    """Spectral line imaging from calibrated (DDE and DIE) data
    
    :param params:
    :return:
    """

    vis = get_parameter(params, 'visibility')
    sm = get_parameter(params, 'skymodel')

    return QA()


def fast_imaging(params):
    """Fast imaging from calibrated (DIE only) data

    :param params:
    :return:
    """
    
    vis = get_parameter(params, 'visibility')
    sm = get_parameter(params, 'skymodel')
    
    return QA()


def EOR(params):
    """EOR calibration and imaging
    
    :param params:
    :return:
    """
    vis = get_parameter(params, 'visibility')
    sm = get_parameter(params, 'skymodel')
    
    return QA()

