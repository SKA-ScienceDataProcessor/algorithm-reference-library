# Tim Cornwell <realtimcornwell@gmail.com>
#
#


from arl_exceptions import *
from arl.visibility_calibration import solve_gains, qa_gaintable
from arl.quality_assessment import QA

def get_parameters(parameters, funcname):
    # Get parameters after checking for existence
    if funcname not in parameters.keys():
        raise ParameterMissing(funcname)
    
    vt = parameters[funcname]['visibility']
    sm = parameters[funcname]['skymodel']
    return vt, sm


def RCAL(parameters):
    """ Real-time calibration: single shot
     
    :param parameters:
    :return:
   """
    funcname='RCAL'
    vt, sm = get_parameters(parameters, funcname)
    
    gains = solve_gains(vt, sm, **(parameters[funcname]))
    qa = qa_gaintable(gains)
    if qa:
        print("pipelines.RCAL: Solution good")
    return qa


def ICAL(parameters):
    """ Post observation image, deconvolve, and self-calibrate
   
    :param parameters:
    :return:
    """
    funcname='ICAL'
    vt, sm = get_parameters(parameters, funcname)
    
    gains = solve_gains(vt, sm, **parameters[funcname])
    qa = qa_gaintable(gains)
    if qa:
        print("pipelines.ICAL: Solution good")
    return qa


def continuum_imaging(**parameters):
    """Continuum imaging from calibrated (DDE and DIE) data

    
    :param parameters:
    :return:
    """

    funcname='continuum_imaging'
    vt, sm = get_parameters(parameters, funcname)
    
    return QA()
    

def spectral_line_imaging(**parameters):
    """Spectral line imaging from calibrated (DDE and DIE) data
    
    :param parameters:
    :return:
    """
    
    funcname='spectral_line_imaging'
    vt, sm = get_parameters(parameters, funcname)

    return QA()


def fast_imaging(parameters):
    funcname = 'fast_imaging'
    vt, sm = get_parameters(parameters, funcname)
    
    return QA()


def EOR(parameters):
    funcname = 'EOR'
    vt, sm = get_parameters(parameters, funcname)
    
    return QA()

