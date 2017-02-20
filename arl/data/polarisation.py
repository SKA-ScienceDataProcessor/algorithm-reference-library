# Tim Cornwell <realtimcornwell@gmail.com>
#
# Polarisation functions
#


class Polarisation_Frame:
    """ Define polarisation frames

    """
    def __init__(self, name):

        self.pol_frames = {
            'circular': {'RR': 0, 'RL': 1, 'LR': 2, 'LL': 3},
            'circularnp': {'RR': 0, 'LL': 1},
            'linear': {'XX': 0, 'XY': 1, 'YX': 2, 'YY': 3},
            'linearnp': {'XX': 0, 'YY': 1},
            'stokesIQUV': {'I': 0, 'Q': 1, 'U': 2, 'V': 3},
            'stokesIV': {'I': 0, 'V': 1},
            'stokesIQ': {'I': 0, 'Q': 1},
            'stokesI':  {'I': 0}
        }
        if name in self.pol_frames.keys():
            self.type = name
            self.translations = self.pol_frames[name]
        else:
            raise RuntimeError("Unknown polarisation frame %s" % str(name))
        
    @property
    def npol(self):
        return len(self.translations.keys())



def convert_stokes_to_linear(inpol, inverse=False):
    return inpol


def convert_stokes_to_circular(inpol, inverse=False):
    return inpol


def convert_linear_to_circular(inpol, inverse=False):
    return inpol


def convert_polarisation_frame(inpol, inframe, outframe):
    return inpol


class MKernel:
    """ Mueller kernel with numpy.array, antenna1, antenna2, time
    
    """


class Jones:
    """ Jones kernel with numpy.array, antenna1, antenna2, time
    """
