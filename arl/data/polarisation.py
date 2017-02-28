# Tim Cornwell <realtimcornwell@gmail.com>
#
# Polarisation functions
#

import numpy


class Receptor_Frame:
    """ Define polarisation frames for receptors

    """
    
    def __init__(self, name):
        
        self.rec_frames = {
            'circular': {'R': 0, 'L': 1},
            'linear': {'X': 0, 'Y': 1},
        }
        if name in self.rec_frames.keys():
            self.type = name
            self.translations = self.rec_frames[name]
        else:
            raise RuntimeError("Unknown receptor frame %s" % str(name))
    
    @property
    def nrec(self):
        """ Number of receptors (should be 2)
        
        """
        return len(self.translations.keys())
    
    def __eq__(self, a):
        return self.type == a.type


class Polarisation_Frame:
    """ Define polarisation frames post correlation

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
        
    def __eq__(self, a):
        return self.type == a.type
        
    @property
    def npol(self):
        """ Number of correlated polarisations"""
        return len(self.translations.keys())


def convert_stokes_to_linear(stokes):
    """ Convert Stokes IQUV to Linear

    :param stokes: [...,4] Stokes vector in I,Q,U,V (can be complex)
    :returns: linear vector in XX, XY, YX, YY sequence

    Equation 4.58 TMS
    """
    conversion_matrix = [[1, 1, 0, 0],
                         [0, 0, 1, 1j],
                         [0, 0, 1, -1j],
                         [1, -1, 0, 0]]

    assert stokes.shape[-1] ==4

    return numpy.dot(conversion_matrix, stokes)


def convert_linear_to_stokes(linear):
    """ Convert Linear to Stokes IQUV

    :param linear: [...,4] linear vector in XX, XY, YX, YY sequence
    :returns: Complex I,Q,U,V

    Equation 4.58 TMS, inverted with numpy.linalg.inv
    """
    conversion_matrix = [[0.5 + 0.j, 0.0 + 0.j, 0.0 + 0.j, 0.5 + 0.j],
                         [0.5 + 0.j, 0.0 + 0.j, 0.0 + 0.j, -0.5 - 0.j],
                         [0.0 + 0.j, 0.5 + 0.j, 0.5 + 0.j, 0.0 + 0.j],
                         [0.0 + 0.j, 0.0 - 0.5j, 0.0 + 0.5j, 0.0 + 0.j]]
    
    assert linear.shape[-1] ==4

    return numpy.dot(conversion_matrix, linear)


def convert_stokes_to_circular(stokes):
    """ Convert Stokes IQUV to Circular

    :param stokes: [...,4] Stokes vector in I,Q,U,V (can be complex)
    :returns: circular vector in RR, RL, LR, LL sequence

    Equation 4.59 TMS
    """
    conversion_matrix = [[1, 0, 0, 1],
                         [0, -1j, 1, 0],
                         [0, -1j, -1, 0],
                         [1, 0, 0, -1]]
    
    assert stokes.shape[-1] ==4

    
    return numpy.dot(conversion_matrix, stokes)

def convert_circular_to_stokes(circular):
    """ Convert Circular to Stokes IQUV

    :param Circular: [...,4] linear vector in RR, RL, LR, LL sequence
    :returns: Complex I,Q,U,V

    Equation 4.58 TMS, inverted with numpy.linalg.inv
    """

    conversion_matrix = [[ 0.5+0.j ,  0.0+0.j ,  0.0+0.j ,  0.5+0.j ],
       [ 0.0+0.j , -0.0+0.5j, -0.0+0.5j,  0.0+0.j ],
       [ 0.0+0.j ,  0.5+0.j , -0.5-0.j ,  0.0+0.j ],
       [ 0.5+0.j ,  0.0+0.j ,  0.0+0.j , -0.5-0.j ]]
    
    assert circular.shape[-1] ==4


    return numpy.dot(conversion_matrix, circular)

def correlate_polarisation(rec_frame: Receptor_Frame):
    """ Gives the polarisation frame corresponding to a receptor frame
    
    :param rec_frame: Receptor frame
    :returns: Polarisation_Frame
    """
    if rec_frame == Receptor_Frame("circular"):
        correlation = Polarisation_Frame("circular")
    elif rec_frame == Receptor_Frame("linear"):
        correlation = Polarisation_Frame("linear")
    else:
        raise RuntimeError("Unknown receptor frame %s for correlation" % rec_frame)
    
    return correlation

def congruent_polarisation(rec_frame: Receptor_Frame, pol_frame: Polarisation_Frame):
    """Are these receptor and polarisation frames congruent?
    
    """
    if rec_frame.type == "linear":
        return pol_frame.type in ["linear", "linearnp"]
    elif rec_frame.type == "circular":
        return pol_frame.type in ["circular", "circularnp"]
    
    return False
