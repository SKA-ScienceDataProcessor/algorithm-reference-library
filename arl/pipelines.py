# Tim Cornwell <realtimcornwell@gmail.com>
#
#


from arl_exceptions import *
from arl.visibility_calibration import solve_gains, qa_gaintable
from arl.quality_assessment import QA
from arl.parameters import get_parameter

def RCAL(parameters):
    """ Real-time calibration: single shot
     
    :param parameters:
    :return:
   """
    vt = get_parameter(parameters, 'visibility', None)
    sm = get_parameter(parameters, 'skymodel', None)

    gains = solve_gains(vt, sm, parameters)
    qa = qa_gaintable(gains)
    if qa:
        print("pipelines.RCAL: Solution good")
    return qa


def ICAL(parameters):
    """ Post observation image, deconvolve, and self-calibrate
   
    :param parameters:
    :return:
    """
    vt = get_parameter(parameters, 'visibility')
    sm = get_parameter(parameters, 'skymodel')

    gains = solve_gains(vt, sm, parameters)
    qa = qa_gaintable(gains)
    if qa:
        print("pipelines.ICAL: Solution good")
    return qa


def continuum_imaging(parameters):
    """Continuum imaging from calibrated (DDE and DIE) data

    
    :param parameters:
    :return:
    """

    vt = get_parameter(parameters, 'visibility')
    sm = get_parameter(parameters, 'skymodel')
    
    return QA()
    

def spectral_line_imaging(parameters):
    """Spectral line imaging from calibrated (DDE and DIE) data
    
    :param parameters:
    :return:
    """

    vt = get_parameter(parameters, 'visibility')
    sm = get_parameter(parameters, 'skymodel')

    return QA()


def fast_imaging(parameters):
    """Fast imaging from calibrated (DIE only) data

    :param parameters:
    :return:
    """
    
    vt = get_parameter(parameters, 'visibility')
    sm = get_parameter(parameters, 'skymodel')
    
    return QA()


def EOR(parameters):
    """EOR calibration and imaging
    
    :param parameters:
    :return:
    """
    vt = get_parameter(parameters, 'visibility')
    sm = get_parameter(parameters, 'skymodel')
    
    return QA()

