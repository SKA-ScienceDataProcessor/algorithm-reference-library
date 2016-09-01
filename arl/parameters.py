""" Parameter handling for ARL functions

Tim Cornwell <realtimcornwell@gmail.com>
"""

import sys

# for current func name, specify 0 or no argument.
# for name of caller of current func, specify 1.
# for name of caller of caller of current func, specify 2. etc.
currentFuncName = lambda n=0: sys._getframe(n + 1).f_code.co_name

def get_parameter(parameters, key, default=None):
    """ Get a specified named value for this (calling) function
    
    The parameter is searched for in parameters and then in parameters[functionname] where is the name of the calling function
    
    :param parameters: Parameter dictionary
    :param key: Key e.g. 'loop_gain'
    :param default: Default value if not specified
    :return: result
    """
    # First look at the top level
    value = default
    if key in parameters.keys():
        value = parameters[key]
    else:
        # Now see if the function name is used
        funcname = currentFuncName(1)
        if funcname in parameters.keys():
            if key in parameters[funcname]:
                value = parameters[funcname]
    
    return value

