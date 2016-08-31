"""Exceptions for ARL

realtimcornwell@gmail.com
"""

class ParameterMissing(BaseException):
    
    def __init__(self, funcname):
        Exception.__init__(self, "Parameters not found for %s" % funcname)