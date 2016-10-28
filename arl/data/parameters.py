"""Parameter handling for ARL functions
"""
#
# Tim Cornwell <realtimcornwell@gmail.com>
#

import logging
import os

log = logging.getLogger("arl.parameters")


def crocodile_path(path):
    """
    Converts a path that might be relative to crocodile root into an
    absolute path.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    crocodile = os.getenv('CROCODILE', project_root)
    return os.path.join(crocodile, path)


def get_parameter(params, key, default=None):
    """ Get a specified named value for this (calling) function
    
    The parameter is searched for in params
    
    :param params: Parameter dictionary
    :param key: Key e.g. 'loop_gain'
    :param default: Default value
    :return: result
    """
    
    value = default
    if key in params.keys():
        value = params[key]
    return value


def import_parameters(paramsfile):
    """Import parameters from a text file
    
    :param paramsfile: file to write to
    :return: parameters as dict
    """
    f = open(paramsfile, 'r')
    d = eval(f.read())
    f.close()
    log.debug("Parameters read")
    log_parameters(d)
    return d


def export_parameters(d, paramsfile):
    """Export parameters to a textfile
    
    :param d: parameters as dict
    :param paramsfile: file to write to
    :return:
    """
    f = open(paramsfile, 'w')
    log.debug("Parameters written")
    log_parameters(d)
    f.write(str(d))
    f.close()


def log_parameters(d):
    for key in d.keys():
        log.debug('log_parameters:   %s      =       %s' % (key, d[key]))
