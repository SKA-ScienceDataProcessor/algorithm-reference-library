# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

"""
Functions that represent polarisation information.
"""
class Stokes:
    """ Stokes representation I,Q,U,V
    
    """
    
class CircularComplexStokes:
    """ Complex Stokes representation
    XX, XY, YX, YY
    
    """


class LinearComplexStokes:
    """ Complex Stokes representation
    XX, XY, YX, YY

    """


class MKernel:
    """ Mueller kernel with numpy.array, antenna1, antenna2, time
    
    """


class Jones:
    """ Jones kernel with numpy.array, antenna1, antenna2, time
    """
