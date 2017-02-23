# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Definition of structures needed by the function interface. These are mostly
subclasses of astropy classes.
"""

from arl.data.data_models import *
from arl.data.parameters import *
from arl.fourier_transforms.ftprocessor_base import invert_2d, predict_2d, predict_skycomponent_visibility, \
    normalize_sumwt
from arl.image.deconvolution import deconvolve_cube
from arl.visibility.operations import copy_visibility

log = logging.getLogger(__name__)


def solve_skymodel(vis: Visibility, model: Image, components=[], predict=predict_2d, invert=invert_2d, **kwargs):
    """Solve for image using deconvolve_cube and specified predict, invert

    This is the same as a majorcycle/minorcycle algorithm. The components are removed prior to deconvolution.

    :param vis:
    :param model: Model image
    :param predict: Predict function e.g. predict_2d, predict_wslice
    :param invert: Invert function e.g. invert_2d, invert_wslice
    :returns: Visibility, model
    """
    nmajor = get_parameter(kwargs, 'nmajor', 5)
    log.info("solve_skymodel: Performing %d major cycles" % nmajor)
    
    # The model is added to each major cycle and then the visibilities are
    # calculated from the full model
    vispred = copy_visibility(vis)
    vispred = predict(vispred, model, **kwargs)
    for sc in components:
        vispred = predict_skycomponent_visibility(vis, sc)
    
    visres = copy_visibility(vispred)
    visres.data['vis'] = vis.data['vis'] - vispred.data['vis']
    dirty, sumwt = invert(visres, model, **kwargs)
    dirty = normalize_sumwt(dirty, sumwt)
    psf, sumwt = invert(visres, model, dopsf=True, **kwargs)
    psf = normalize_sumwt(psf, sumwt)
    
    thresh = get_parameter(kwargs, "threshold", 0.0)
    
    for i in range(nmajor):
        log.info("solve_skymodel: Start of major cycle %d" % i)
        cc, res = deconvolve_cube(dirty, psf, **kwargs)
        model.data += cc.data
        vispred = predict_2d(vispred, model, **kwargs)
        visres.data['vis'] = vis.data['vis'] - vispred.data['vis']
        dirty, sumwt = invert_2d(visres, model, **kwargs)
        if numpy.abs(dirty.data).max() < 1.1 * thresh:
            log.info("Reached stopping threshold %.6f Jy" % thresh)
            break
        log.info("solve_skymodel: End of major cycle")
    log.info("solve_skymodel: End of major cycles")
    return visres, model, dirty
