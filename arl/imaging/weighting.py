"""
Functions that aid weighting the visibility data prior to imaging.

There are two classes of functions:
    - Changing the weight dependent on noise level or sample density or a combination
    - Tapering the weihght spatially to avoid effects of sharp edges or to emphasize a given scale size in the image

"""

import numpy

from arl.data.data_models import Visibility, Image
from arl.data.parameters import get_parameter
from arl.fourier_transforms.convolutional_gridding import weight_gridding
from arl.imaging import get_polarisation_map, get_uvw_map
from arl.imaging.params import get_frequency_map


def weight_visibility(vis: Visibility, im: Image, **kwargs) -> Visibility:
    """ Reweight the visibility data using a selected algorithm

    Imaging uses the column "imaging_weight" when imaging. This function sets that column using a
    variety of algorithms
    
    Options are:
        - Natural: by visibility weight (optimum for noise in final image)
        - Uniform: weight of sample divided by sum of weights in cell (optimum for sidelobes)
        - Super-uniform: As uniform, by sum of weights is over extended box region
        - Briggs: Compromise between natural and uniform
        - Super-briggs: As Briggs, by sum of weights is over extended box region

    :param vis:
    :param im:
    :return: visibility with imaging_weights column added and filled
    """
    assert isinstance(vis, Visibility), "vis is not a Visibility: %r" % vis
    
    assert get_parameter(kwargs, "padding", False) is False
    spectral_mode, vfrequencymap = get_frequency_map(vis, im)
    polarisation_mode, vpolarisationmap = get_polarisation_map(vis, im)
    uvw_mode, shape, padding, vuvwmap = get_uvw_map(vis, im)
    
    density = None
    densitygrid = None
    
    weighting = get_parameter(kwargs, "weighting", "uniform")
    vis.data['imaging_weight'], density, densitygrid = weight_gridding(im.data.shape, vis.data['weight'], vuvwmap,
                                                                       vfrequencymap, vpolarisationmap, weighting)
    
    return vis, density, densitygrid


def taper_visibility_gaussian(vis: Visibility, beam=None) -> Visibility:
    """ Taper the visibility weights

    These are cumulative. If You can reset the imaging_weights
    using :py:mod:`arl.imaging.weighting.weight_visibility`

    :param vis: Visibility with imaging_weight's to be tapered
    :param beam: desired resolution (Full width half maximum, radians)
    :return: visibility with imaging_weight column modified
    """
    assert isinstance(vis, Visibility), "vis is not a Visibility: %r" % vis

    if beam is None:
        raise ValueError("Beam size not specified for Gaussian taper")
    uvdistsq = vis.u ** 2 + vis.v ** 2
    # See http://mathworld.wolfram.com/FourierTransformGaussian.html
    scale_factor = numpy.pi ** 2 * beam ** 2 / (4.0 * numpy.log(2.0))
    wt = numpy.exp(-scale_factor * uvdistsq)
    for row in range(vis.nvis):
        vis.data['imaging_weight'][row, ...] = vis.imaging_weight[row, ...] * wt[row]

    return vis


def taper_visibility_tukey(vis: Visibility, tukey=0.1) -> Visibility:
    """ Taper the visibility weights
    
    This algorithm is present in WSClean.

    See https://sourceforge.net/p/wsclean/wiki/Tapering

    tukey, a circular taper that smooths the outer edge set by -maxuv-l
    inner-tukey, a circular taper that smooths the inner edge set by -minuv-l
    edge-tukey, a square-shaped taper that smooths the edge set by the uv grid and -taper-edge.

    These are cumulative. If You can reset the imaging_weights
    using :py:mod:`arl.imaging.weighting.weight_visibility`

    :param vis: Visibility with imaging_weight's to be tapered
    :return: visibility with imaging_weight column modified
    """
    assert isinstance(vis, Visibility), "vis is not a Visibility: %r" % vis
    
    uvdist = numpy.sqrt(vis.u ** 2 + vis.v ** 2)
    uvdistmax = numpy.max(uvdist)
    uvdist /= uvdistmax
    wt = numpy.array([tukey_filter(uv, tukey) for uv in uvdist])
    for row in range(vis.nvis):
        vis.data['imaging_weight'][row, ...] = vis.imaging_weight[row, ...] * wt[row]
   
    return vis


def tukey_filter(x, r):
    """ Calculate the Tukey (tapered cosine) filter
    
    See e.g. https://uk.mathworks.com/help/signal/ref/tukeywin.html

    :param x: x coordinate (float)
    :param r: transition point of filter (float)
    :returns: Value of filter for x
    """
    if x >= 0.0 and x < r / 2.0:
        return 0.5 * (1.0 + numpy.cos(2.0 * numpy.pi * (x - r / 2.0) / r))
    elif x >= 1 - r / 2.0 and x <= 1.0:
        return 0.5 * (1.0 + numpy.cos(2.0 * numpy.pi * (x - 1 + r / 2.0) / r))
    else:
        return 1.0
