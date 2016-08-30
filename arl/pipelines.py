# Tim Cornwell <realtimcornwell@gmail.com>
#
#

from arl.visibility_calibration import solve_gains, qa_gaintable, create_visibility
from arl.fourier_transforms import predict_visibility
from arl.quality_assessment import QA

def RCAL(parameters, logfile=None):
    """ Real-time calibration

    """
    print(parameters)
    vt = parameters['RCAL']['visibility']
    sm = parameters['RCAL']['skymodel']
    
    gains = solve_gains(vt, sm, **parameters)
    qa = qa_gaintable(gains)
    if qa:
        print("pipelines.RCAL: Solution good")
    return qa


def ICAL(parameters, logfile=None):
    """ Self-calibration

    """
    print(parameters)
    vt = parameters['ICAL']['visibility']
    sm = parameters['ICAL']['skymodel']
    
    gains = solve_gains(vt, sm, **parameters)
    qa = qa_gaintable(gains)
    if qa:
        print("pipelines.RCAL: Solution good")
    return qa


def continuum_imaging(parameters, logfile=None):

    print(parameters)
    vt = parameters['continuum_imaging']['visibility']
    sm = parameters['continuum_imaging']['skymodel']
    
    return QA()
    

def spectral_line_imaging(parameters, logfile=None):
    
    print(parameters)
    vt = parameters['spectral_line_imaging']['visibility']
    sm = parameters['spectral_line_imaging']['skymodel']
    
    return QA()


            