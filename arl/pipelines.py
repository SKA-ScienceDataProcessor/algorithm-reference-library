""" Pipeline definitions

"""
# Tim Cornwell <realtimcornwell@gmail.com>
#
#

from data.parameters import *
from util.quality_assessment import QA
from visibility.visibility_calibration import solve_gains, qa_gaintable

log = logging.getLogger("arl.pipelines")

def RCAL(params):
    """ Real-time calibration: single shot
     
    :param params: Dictionary containing parameters
    :return:
   """
    # TODO: implement

    log_parameters(params)
    vis = get_parameter(params, 'visibility', None)
    sm = get_parameter(params, 'skymodel', None)

    gains = solve_gains(vis, sm, params)
    qa = qa_gaintable(gains)
    if qa:
        log.info("RCAL: Solution good")
    return qa


def ICAL(params):
    """ Post observation image, deconvolve, and self-calibrate
   
    :param params: Dictionary containing parameters
    :return:
    """
    # TODO: implement

    log_parameters(params)
    vis = get_parameter(params, 'visibility')
    sm = get_parameter(params, 'skymodel')

    gains = solve_gains(vis, sm, params)
    qa = qa_gaintable(gains)
    if qa:
        log.info("ICAL: Solution good")
    return qa


def continuum_imaging(params):
    """Continuum imaging from calibrated (DDE and DIE) data

    
    :param params: Dictionary containing parameters
    :return:
    """

    # TODO: implement

    log_parameters(params)
    vis = get_parameter(params, 'visibility')
    sm = get_parameter(params, 'skymodel')
    
    return QA()
    

def spectral_line_imaging(params):
    """Spectral line imaging from calibrated (DDE and DIE) data
    
    :param params: Dictionary containing parameters
    :return:
    """
    # TODO: implement

    log_parameters(params)
    vis = get_parameter(params, 'visibility')
    sm = get_parameter(params, 'skymodel')

    return QA()


def fast_imaging(params):
    """Fast imaging from calibrated (DIE only) data

    :param params: Dictionary containing parameters
    :return:
    """
    # TODO: implement

    log_parameters(params)
    vis = get_parameter(params, 'visibility')
    sm = get_parameter(params, 'skymodel')
    
    return QA()


def EOR(params):
    """EOR calibration and imaging
    
    :param params: Dictionary containing parameters
    :return:
    """
    # TODO: implement

    log_parameters(params)
    vis = get_parameter(params, 'visibility')
    sm = get_parameter(params, 'skymodel')
    
    return QA()

