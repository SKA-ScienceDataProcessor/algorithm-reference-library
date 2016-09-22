# Parameter handling for ARL functions
#
# Tim Cornwell <realtimcornwell@gmail.com>
#

import sys

import logging
log = logging.getLogger( "arl.parameters" )

def get_parameter(params, key, default=None, level=1):
    """ Get a specified named value for this (calling) function
    
    The parameter is searched for in params and then in params[functionname] where is the name of the calling function
    
    :param params: Parameter dictionary
    :param key: Key e.g. 'loop_gain'
    :param default: Default value
    :param level: Level in stack for function name: 0: self, 1 parent, 2 parent of parent
    :return: result
    """
    # for current func name, specify 0 or no argument.
    # for name of caller of current func, specify 1.
    # for name of caller of caller of current func, specify 2. etc.
    currentFuncName = lambda n=0: sys._getframe(n + 1).f_code.co_name

    # First look at the top level
    value = default
    if key in params.keys():
        value = params[key]

    # Now see if the function name is used
    funcname = currentFuncName(level)
    if funcname in params.keys():
        if key in params[funcname]:
            value = params[funcname]
    
    return value

def import_parameters(paramsfile):
    """Import parameters from a text file
    
    :param paramsfile: file to write to
    :return: parameters as dict
    """
    f = open(paramsfile, 'r')
    d = eval(f.read())
    f.close()
    log.info("Parameters read")
    log_parameters(d)
    return d

def export_parameters(d, paramsfile):
    """Export parameters to a textfile
    
    :param d: parameters as dict
    :param paramsfile: file to write to
    :return:
    """
    f = open(paramsfile, 'w')
    log.info("Parameters written")
    log_parameters(d)
    f.write(str(d))
    f.close()
    
def log_parameters(d):
    for key in d.keys():
        log.info('  %s  =   %s' % (key, d[key]))
    
