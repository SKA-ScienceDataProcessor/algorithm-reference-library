""" Functions for defining polarisation conventions. These include definitions via classes and
    conversion functions.

For example::

    stokes = numpy.array(random.uniform(-1.0, 1.0, [3, 4, 128, 128]))
    ipf = PolarisationFrame('stokesIQUV')
    opf = PolarisationFrame('circular')
    cir = convert_pol_frame(stokes, ipf, opf)
    st = convert_pol_frame(cir, opf, ipf)

or::

    stokes = numpy.array([1, 0.5, 0.2, -0.1])
    circular = convert_stokes_to_circular(stokes)
    
These function operate on Numpy arrays. These are packaged for use in Images. The Image functions are
probably more useful.

"""

import numpy

import logging

log = logging.getLogger(__name__)


class ReceptorFrame:
    """Define polarisation frames for receptors

    circular, linear, and stokesI. The latter is non-physical but useful for some types of testing.
    """

    rec_frames = {
        'circular': {'R': 0, 'L': 1},
        'linear': {'X': 0, 'Y': 1},
        'stokesI': {'I': 0}
    }

    def __init__(self, name):
        """ create ReceptorFrame

        :param name:
        """

        if name in self.rec_frames.keys():
            self.type = name
            self.translations = self.rec_frames[name]
        else:
            raise ValueError("Unknown receptor frame %s" % str(name))

    @property
    def nrec(self):
        """ Number of receptors (should be 2)
        """
        return len(list(self.translations.keys()))

    def valid(self, name):
        return name in self.rec_frames.keys()

    def __eq__(self, a):
        return self.type == a.type


class PolarisationFrame:
    """ Define polarisation frames post correlation

    """
    fits_codes = {
        'circular': [-1, -2, -3, -4],
        'circularnp': [-1, -4],
        'linear': [-5, -6, -7, -8],
        'linearnp': [-5, -8],
        'stokesIQUV': [1, 2, 3, 4],
        'stokesIV': [1, 4],
        'stokesIQ': [1, 2],
        'stokesI': [1]
    }
    polarisation_frames = {
        'circular': {'RR': 0, 'RL': 1, 'LR': 2, 'LL': 3},
        'circularnp': {'RR': 0, 'LL': 1},
        'linear': {'XX': 0, 'XY': 1, 'YX': 2, 'YY': 3},
        'linearnp': {'XX': 0, 'YY': 1},
        'stokesIQUV': {'I': 0, 'Q': 1, 'U': 2, 'V': 3},
        'stokesIV': {'I': 0, 'V': 1},
        'stokesIQ': {'I': 0, 'Q': 1},
        'stokesI': {'I': 0}
    }

    def __init__(self, name):
        """ create PolarisationFrame

        :param name:
        """

        if name in self.polarisation_frames.keys():
            self.type = name
            self.translations = self.polarisation_frames[name]
        else:
            raise ValueError("Unknown polarisation frame %s" % str(name))

    def __eq__(self, a):
        if a is None:
            return False
        return self.type == a.type

    @property
    def npol(self):
        """ Number of correlated polarisations
        """
        return len(list(self.translations.keys()))


def polmatrixmultiply(cm, vec, polaxis=1):
    """Matrix multiply of appropriate axis of vec [...,:] by cm

    For an image vec has axes [nchan, npol, ny, nx] and polaxis=1
    For visibility vec has axes [row, nchan, npol] and polaxis=2

    :param cm: matrix to apply
    :param vec: array to be multiplied [...,:]
    :param polaxis: which axis contains the polarisation
    :return: multiplied vec
    """
    if len(vec.shape) == 1:
        return numpy.dot(cm, vec)
    else:
        # This tensor swaps the first two axes so we need to tranpose back
        result = numpy.tensordot(cm, vec, axes=(1, polaxis))
        permut = list(range(len(result.shape)))
        permut[0], permut[polaxis] = permut[polaxis], permut[0]
        return numpy.transpose(result, axes=permut)


def convert_stokes_to_linear(stokes, polaxis=1):
    """ Convert Stokes IQUV to Linear

    :param stokes: [...,4] Stokes vector in I,Q,U,V (can be complex)
    :param polaxis: Axis of stokes with polarisation (default 1)
    :return: linear vector in XX, XY, YX, YY sequence

    Equation 4.58 TMS
    """
    conversion_matrix = numpy.array([[1, 1, 0, 0],
                                     [0, 0, 1, 1j],
                                     [0, 0, 1, -1j],
                                     [1, -1, 0, 0]])

    return polmatrixmultiply(conversion_matrix, stokes, polaxis)


def convert_linear_to_stokes(linear, polaxis=1):
    """ Convert Linear to Stokes IQUV

    :param linear: [...,4] linear vector in XX, XY, YX, YY sequence
    :param polaxis: Axis of linear with polarisation (default 1)
    :return: Complex I,Q,U,V

    Equation 4.58 TMS, inverted with numpy.linalg.inv
    """
    conversion_matrix = numpy.array([[0.5 + 0.j, 0.0 + 0.j, 0.0 + 0.j, 0.5 + 0.j],
                                     [0.5 + 0.j, 0.0 + 0.j, 0.0 + 0.j, -0.5 - 0.j],
                                     [0.0 + 0.j, 0.5 + 0.j, 0.5 + 0.j, 0.0 + 0.j],
                                     [0.0 + 0.j, 0.0 - 0.5j, 0.0 + 0.5j, 0.0 + 0.j]])

    return polmatrixmultiply(conversion_matrix, linear, polaxis)


def convert_stokes_to_circular(stokes, polaxis=1):
    """ Convert Stokes IQUV to Circular

    :param stokes: [...,4] Stokes vector in I,Q,U,V (can be complex)
    :param polaxis: Axis of stokes with polarisation (default 1)
    :return: circular vector in RR, RL, LR, LL sequence

    Equation 4.59 TMS
    """
    conversion_matrix = numpy.array([[1, 0, 0, 1],
                                     [0, -1j, 1, 0],
                                     [0, -1j, -1, 0],
                                     [1, 0, 0, -1]])

    return polmatrixmultiply(conversion_matrix, stokes, polaxis)


def convert_circular_to_stokes(circular, polaxis=1):
    """ Convert Circular to Stokes IQUV

    :param circular: [...,4] linear vector in RR, RL, LR, LL sequence
    :param polaxis: Axis of circular with polarisation (default 1)
    :return: Complex I,Q,U,V

    Equation 4.58 TMS, inverted with numpy.linalg.inv
    """

    conversion_matrix = numpy.array([[0.5 + 0.j, 0.0 + 0.j, 0.0 + 0.j, 0.5 + 0.j],
                                     [0.0 + 0.j, -0.0 + 0.5j, -0.0 + 0.5j, 0.0 + 0.j],
                                     [0.0 + 0.j, 0.5 + 0.j, -0.5 - 0.j, 0.0 + 0.j],
                                     [0.5 + 0.j, 0.0 + 0.j, 0.0 + 0.j, -0.5 - 0.j]])

    return polmatrixmultiply(conversion_matrix, circular, polaxis)


def convert_pol_frame(polvec, ipf: PolarisationFrame, opf: PolarisationFrame, polaxis=1):
    if ipf == opf:
        return polvec

    if ipf == PolarisationFrame("linear"):
        if opf == PolarisationFrame("stokesIQUV"):
            return convert_linear_to_stokes(polvec, polaxis)
        else:
            raise ValueError("Unknown polarisation conversion")

    if ipf == PolarisationFrame("circular"):
        if opf == PolarisationFrame("stokesIQUV"):
            return convert_circular_to_stokes(polvec, polaxis)
        else:
            raise ValueError("Unknown polarisation conversion")

    if ipf == PolarisationFrame("stokesIQUV"):
        if opf == PolarisationFrame("linear"):
            return convert_stokes_to_linear(polvec, polaxis)
        elif opf == PolarisationFrame("circular"):
            return convert_stokes_to_circular(polvec, polaxis)
        else:
            raise ValueError("Unknown polarisation conversion")

    if ipf == PolarisationFrame("stokesI"):
        if opf == PolarisationFrame("stokesI"):
            return polvec

    raise ValueError("Unknown polarisation conversion")


def correlate_polarisation(rec_frame: ReceptorFrame):
    """ Gives the polarisation frame corresponding to a receptor frame

    :param rec_frame: Receptor frame
    :return: PolarisationFrame
    """
    if rec_frame == ReceptorFrame("circular"):
        correlation = PolarisationFrame("circular")
    elif rec_frame == ReceptorFrame("linear"):
        correlation = PolarisationFrame("linear")
    elif rec_frame == ReceptorFrame("stokesI"):
        correlation = PolarisationFrame("stokesI")
    else:
        raise ValueError("Unknown receptor frame %s for correlation" % rec_frame)

    return correlation


def congruent_polarisation(rec_frame: ReceptorFrame, polarisation_frame: PolarisationFrame):
    """Are these receptor and polarisation frames congruent?

    """
    if rec_frame.type == "linear":
        return polarisation_frame.type in ["linear", "linearnp"]
    elif rec_frame.type == "circular":
        return polarisation_frame.type in ["circular", "circularnp"]
    elif rec_frame.type == "stokesI":
        return polarisation_frame.type == "stokesI"

    return False
