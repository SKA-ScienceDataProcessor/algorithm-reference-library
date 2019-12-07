"""
Functions that aid weighting the visibility data prior to imaging.

There are two classes of functions:
    - Changing the weight dependent on noise level or sample density or a combination
    - Tapering the weihght spatially to avoid effects of sharp edges or to emphasize a given scale size in the image

"""

__all__ = ['weight_visibility', 'taper_visibility_gaussian', 'taper_visibility_tukey']


import numpy

from data_models.memory_data_models import Visibility, BlockVisibility
from processing_components.griddata.gridding import grid_weight_to_griddata, griddata_reweight
from processing_components.griddata.kernels import create_pswf_convolutionfunction
from processing_components.griddata.operations import create_griddata_from_image
from processing_library.util.array_functions import tukey_filter


def weight_visibility(vis, model, gcfcf=None, weighting='uniform', **kwargs):
    """ Weight the visibility data

    This is done collectively so the weights are summed over all vis_lists and then
    corrected

    :param vis_list:
    :param model_imagelist: Model required to determine weighting parameters
    :param weighting: Type of weighting
    :param kwargs: Parameters for functions in graphs
    :return: List of vis_graphs
   """
    
    assert isinstance(vis, Visibility), vis

    if gcfcf is None:
        gcfcf = create_pswf_convolutionfunction(model)
    
    griddata = create_griddata_from_image(model)
    griddata, sumwt = grid_weight_to_griddata(vis, griddata, gcfcf[1])
    vis = griddata_reweight(vis, griddata, gcfcf[1])
    return vis


def taper_visibility_gaussian(vis: Visibility, beam=None) -> Visibility:
    """ Taper the visibility weights

    These are cumulative. If You can reset the imaging_weights
    using :py:mod:`processing_library.imaging.weighting.weight_visibility`

    :param vis: Visibility with imaging_weight's to be tapered
    :param beam: desired resolution (Full width half maximum, radians)
    :return: visibility with imaging_weight column modified
    """
    assert isinstance(vis, Visibility), vis
    
    if beam is None:
        raise ValueError("Beam size not specified for Gaussian taper")
    uvdistsq = vis.u ** 2 + vis.v ** 2
    # See http://mathworld.wolfram.com/FourierTransformGaussian.html
    scale_factor = numpy.pi ** 2 * beam ** 2 / (4.0 * numpy.log(2.0))
    prior = vis.imaging_weight[:, :]
    wt = numpy.exp(-scale_factor * uvdistsq)
    vis.data['imaging_weight'][:, :] = vis.imaging_weight[:, :] * wt[:, numpy.newaxis]
    
    return vis


def taper_visibility_tukey(vis: Visibility, tukey=0.1) -> Visibility:
    """ Taper the visibility weights
    
    This algorithm is present in WSClean.

    See https://sourceforge.net/p/wsclean/wiki/Tapering

    tukey, a circular taper that smooths the outer edge set by -maxuv-l
    inner-tukey, a circular taper that smooths the inner edge set by -minuv-l
    edge-tukey, a square-shaped taper that smooths the edge set by the uv grid and -taper-edge.

    These are cumulative. If You can reset the imaging_weights
    using :py:mod:`processing_library.imaging.weighting.weight_visibility`

    :param vis: Visibility with imaging_weight's to be tapered
    :return: visibility with imaging_weight column modified
    """

    assert isinstance(vis, Visibility), vis

    uvdist = numpy.sqrt(vis.u ** 2 + vis.v ** 2)
    uvdistmax = numpy.max(uvdist)
    uvdist /= uvdistmax
    wt = numpy.array([tukey_filter(uv, tukey) for uv in uvdist])
    vis.data['imaging_weight'][:, :] = vis.imaging_weight[:, :] * wt[:, numpy.newaxis]
    
    return vis

