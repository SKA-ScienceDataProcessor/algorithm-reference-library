""" Pipeline functionss

"""
# Tim Cornwell <realtimcornwell@gmail.com>
#
#

from arl.data.parameters import *

log = logging.getLogger(__name__)

def RCAL(**kwargs):
    """ Real-time calibration: single shot
     
    :param params: Dictionary containing parameters
    :return:
   """
    # TODO: implement

    log_parameters(**kwargs)
    return True

def ICAL(**kwargs):
    """ Post observation image, deconvolve, and self-calibrate
   
    :param params: Dictionary containing parameters
    :return:
    """
    # TODO: implement

    log_parameters(**kwargs)
    return True

def continuum_imaging(**kwargs):
    """Continuum imaging from calibrated (DDE and DIE) data

    
    :param params: Dictionary containing parameters
    :return:
    """

    # TODO: implement

    log_parameters(**kwargs)
    return True
    

def spectral_line_imaging(**kwargs):
    """Spectral line imaging from calibrated (DDE and DIE) data
    
    :param params: Dictionary containing parameters
    :return:
    """
    # TODO: implement

    log_parameters(**kwargs)
    return True


def fast_imaging(**kwargs):
    """Fast imaging from calibrated (DIE only) data

    :param params: Dictionary containing parameters
    :return:
    """
    # TODO: implement

    log_parameters(**kwargs)
    return True


def EOR(**kwargs):
    """EOR calibration and imaging
    
    :param params: Dictionary containing parameters
    :return:
    """
    # TODO: implement

    log_parameters(**kwargs)
    return True

