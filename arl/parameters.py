# Parameter handling for ARL functions
#
# Tim Cornwell <realtimcornwell@gmail.com>
#

import sys

# for current func name, specify 0 or no argument.
# for name of caller of current func, specify 1.
# for name of caller of caller of current func, specify 2. etc.
currentFuncName = lambda n=0: sys._getframe(n + 1).f_code.co_name

def get_parameter(params, key, default=None, level=1):
    """ Get a specified named value for this (calling) function
    
    The parameter is searched for in params and then in params[functionname] where is the name of the calling function
    
    :param params: Parameter dictionary
    :param key: Key e.g. 'loop_gain'
    :param default: Default value if not specified
    "param level: Level in stack: 1 parent, 2 parent of parent
    :return: result
    """
    # First look at the top level
    value = default
    if key in params.keys():
        value = params[key]
    else:
        # Now see if the function name is used
        funcname = currentFuncName(level)
        if funcname in params.keys():
            if key in params[funcname]:
                value = params[funcname]
    
    return value

