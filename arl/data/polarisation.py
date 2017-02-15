# Tim Cornwell <realtimcornwell@gmail.com>
#
# Polarisation functions
#


class Polarisation_Frame:
    """ Define polarisation frames

    """
    
    @property
    def circular(self):
        return 'circular', {'RR': 0, 'RL': 1, 'LR': 2, 'LL': 3}

    @staticmethod
    def circularnp():
        return 'circularnp', {'RR': 0, 'LL': 3}

    @property
    def linear(self):
        return 'linear', {'XX': 0, 'XY': 1, 'YX': 2, 'YY': 3}

    @property
    def linearnp(self):
        return 'linearnp', {'XX': 0, 'YY': 3}

    @property
    def stokes(self):
        return 'stokes', {'I': 0, 'Q': 1, 'U': 2, 'V': 3}

    @property
    def stokesIV(self):
        return 'stokes', {'I': 0, 'V': 3}

    @property
    def stokesIQ(self):
        return 'stokes', {'I': 0, 'Q': 1}
    
    @property
    def stokesI(self):
        return 'stokes', {'I': 0}


"""
Functions that represent polarisation information.
"""
def convert_stokes_to_linear(inpol, inverse=False):
    return inpol

def convert_stokes_to_circular(inpol, inverse=False):
    return inpol

def convert_linear_to_circular(inpol, inverse=False):
    return inpol

def convert_polarisation_frame(inpol, inframe=Polarisation_Frame.stokes, outframe=Polarisation_Frame.linear):
    return inpol


class MKernel:
    """ Mueller kernel with numpy.array, antenna1, antenna2, time
    
    """


class Jones:
    """ Jones kernel with numpy.array, antenna1, antenna2, time
    """
