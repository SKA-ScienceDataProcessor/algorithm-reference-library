"""Parameter handling for ARL functions
"""
#
# Tim Cornwell <realtimcornwell@gmail.com>
#

import logging
import os

log = logging.getLogger(__name__)


def arl_path(path):
    """
    Converts a path that might be relative to ARL root into an
    absolute path.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    arlhome = os.getenv('ARL', project_root)
    return os.path.join(arlhome, path)


def get_parameter(kwargs, key, default=None):
    """ Get a specified named value for this (calling) function
    
    The parameter is searched for in params
    
    :param params: Parameter dictionary
    :param key: Key e.g. 'loop_gain'
    :param default: Default value
    :return: result
    """
    
    if kwargs is None:
        return default
    
    value = default
    if key in kwargs.keys():
        value = kwargs[key]
    return value

def log_parameters(**kwargs):
    if kwargs is not None:
        for key in kwargs.keys():
            log.debug('log_parameters:   %s      =       %s' % (key, kwargs[key]))
