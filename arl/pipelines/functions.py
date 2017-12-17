""" Pipeline functions

"""
#
#
import collections
import logging
import numpy

from arl.data.data_models import Image, Visibility, BlockVisibility, GainTable
from arl.data.parameters import get_parameter
from arl.calibration.solvers import solve_gaintable
from arl.calibration.operations import apply_gaintable
from arl.image.solvers import solve_image
from arl.image.deconvolution import deconvolve_cube, restore_cube
from arl.visibility.base import copy_visibility
from arl.imaging import predict_2d, invert_2d, predict_skycomponent_blockvisibility, \
    predict_skycomponent_visibility

log = logging.getLogger(__name__)


def rcal(vis: BlockVisibility, components, **kwargs) -> GainTable:
    """ Real-time calibration pipeline.
    
    Reads visibilities through a BlockVisibility iterator, calculates model visibilities according to a
    component-based sky model, and performs calibration solution, writing a gaintable for each chunk of
    visibilities.
    
    :param vis: Visibility or Union(Visibility, Iterable)
    :param components: Component-based sky model
    :param kwargs: Parameters
    :return: gaintable
   """
    
    if not isinstance(vis, collections.Iterable):
        vis = [vis]
    
    for ichunk, vischunk in enumerate(vis):
        vispred = copy_visibility(vischunk, zero=True)
        vispred = predict_skycomponent_blockvisibility(vispred, components)
        gt = solve_gaintable(vischunk, vispred, **kwargs)
        yield gt


def ical(vis: BlockVisibility, model: Image, components=None, predict=predict_2d,
         invert=invert_2d, **kwargs) -> (BlockVisibility, Image, Image):
    """ Post observation image, deconvolve, and self-calibrate
   
    :param vis:
    :param model: Model image
    :param predict: Predict function e.g. predict_2d, predict_wstack
    :param invert: Invert function e.g. invert_2d, invert_wstack
    :return: Visibility, model, residual
    """
    nmajor = get_parameter(kwargs, 'nmajor', 5)
    log.info("ical: Performing %d major cycles" % nmajor)
    first_selfcal = get_parameter(kwargs, "first_selfcal", None)
    
    # The model is added to each major cycle and then the visibilities are
    # calculated from the full model
    vispred = copy_visibility(vis)
    visres = copy_visibility(vis)

    vispred = predict(vispred, model, **kwargs)

    if components is not None:
        vispred = predict_skycomponent_blockvisibility(vispred, components)
        
    doselfcal = first_selfcal is not None and first_selfcal == 0
    if doselfcal:
        gt = solve_gaintable(vis, vispred, **kwargs)
        vis = apply_gaintable(vis, gt, inverse=True)

    visres.data['vis'] = vis.data['vis'] - vispred.data['vis']
    dirty, sumwt = invert(visres, model, **kwargs)
    psf, sumwt = invert(visres, model, dopsf=True, **kwargs)

    thresh = get_parameter(kwargs, "threshold", 0.0)

    for i in range(nmajor):
        log.info("ical: Start of major cycle %d" % i)
        cc, res = deconvolve_cube(dirty, psf, **kwargs)
        model.data += cc.data
        vispred = predict(vispred, model, **kwargs)
        doselfcal = first_selfcal is not None and i == first_selfcal
        if doselfcal:
            log.info("Performing selcalibration")
            gt = solve_gaintable(vis, vispred, **kwargs)
            vis = apply_gaintable(vis, gt, inverse=True)
            
        visres.data['vis'] = vis.data['vis'] - vispred.data['vis']
        
        dirty, sumwt = invert(visres, model, **kwargs)
        if numpy.abs(dirty.data).max() < 1.1 * thresh:
            log.info("Reached stopping threshold %.6f Jy" % thresh)
            break
        log.info("solve_image: End of major cycle")

    log.info("solve_image: End of major cycles")
    return visres, model, dirty


def continuum_imaging(vis: Visibility, model: Image, components=None,
                      predict=predict_2d, invert=invert_2d,**kwargs) -> (Image, Image, Image):
    """Continuum imaging from calibrated (DDE and DIE) and coalesced data

    The model image is used as the starting point, and also to determine the imagesize and sampling. Components
    are subtracted before deconvolution.
    
    Uses :py:func:`arl.image.solvers.solve_image`
    
    :param vis: Visibility
    :param model: model image
    :param components: Component-based sky model
    :param kwargs: Parameters
    :return:
    """
    kwargs['first_selfcal'] = None
    return ical(vis, model, components=None, predict=predict, invert=invert, **kwargs)


def spectral_line_imaging(vis: Visibility, model: Image, continuum_model: Image=None, continuum_components=None,
                          predict=predict_2d, invert=invert_2d, deconvolve_spectral=False,
                          **kwargs) -> (Image, Image, Image):
    """Spectral line imaging from calibrated (DIE) data
    
    A continuum model can be subtracted, and deconvolution is optional.
    
    If deconvolve_spectral is True then the solve_image is used to deconvolve.
    If deconvolve_spectral is False then the residual image after continuum subtraction is calculated
    
    :param vis: Visibility
    :param continuum_model: model continuum image to be subtracted
    :param continuum_components: mode components to be subtracted
    :param spectral_model: model spectral image
    :param predict: Predict fumction e.g. predict_2d
    :param invert: Invert function e.g. invert_wprojection
    :return: Residual visibility, spectral model image, spectral residual image
    """

    vis_no_continuum = copy_visibility(vis)
    if continuum_model is not None:
        vis_no_continuum = predict(vis_no_continuum, continuum_model, **kwargs)
    if continuum_components is not None:
        vis_no_continuum = predict_skycomponent_visibility(vis_no_continuum, continuum_components)
    vis_no_continuum.data['vis'] = vis.data['vis'] - vis_no_continuum.data['vis']
    
    if deconvolve_spectral:
        log.info("spectral_line_imaging: Deconvolving continuum subtracted visibility")
        vis_no_continuum, spectral_model, spectral_residual = solve_image(vis_no_continuum,
                                                                          model, **kwargs)
    else:
        log.info("spectral_line_imaging: Making dirty image from continuum subtracted visibility")
        spectral_model, spectral_residual = \
            invert(vis_no_continuum, model, **kwargs)
    
    return vis_no_continuum, spectral_model, spectral_residual


def fast_imaging(**kwargs) -> (Image, Image, Image):
    """Fast imaging from calibrated (DIE only) data

    :param kwargs: Dictionary containing parameters
    :return:
    """
    # TODO: implement
    
    return True


def eor(**kwargs) -> (Image, Image, Image):
    """eor calibration and imaging
    
    :param kwargs: Dictionary containing parameters
    :return:
    """
    # TODO: implement
    
    return True
