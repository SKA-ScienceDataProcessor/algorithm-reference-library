""" Pipelines expressed as dask bags
"""

import logging

import numpy
from dask import bag

from arl.data.parameters import get_parameter
from arl.graphs.bags import deconvolve_bag, invert_bag, predict_bag, residual_image_bag, \
    restore_bag, calibrate_bag, reify, map_record, residual_vis_bag
from arl.visibility.coalesce import convert_blockvisibility_to_visibility

log = logging.getLogger(__name__)


def continuum_imaging_pipeline_bag(vis_bag, model_bag, context, **kwargs) -> bag:
    """ Create bag for the continuum imaging pipeline.
    
    Same as ICAL but with no selfcal.
    
    :param vis_bag: Bag containing BlockVisibility's
    :param model_bag: Image model in a bag
    :param context: Imaging context
    :param kwargs: Parameters for functions in bags
    :return:
    """
    assert isinstance(vis_bag, bag.Bag), vis_bag
    assert isinstance(model_bag, bag.Bag), model_bag
    
    return ical_pipeline_bag(vis_bag, model_bag, context=context, first_selfcal=numpy.infty, **kwargs)


def spectral_line_imaging_pipeline_bag(vis_bag, model_bag,
                                       continuum_model_bag=None,
                                       context='2d',
                                       **kwargs) -> bag:
    """Create bag for spectral line imaging pipeline

    Uses the ical pipeline after subtraction of a continuum model
    
    :param vis_bag: Bag containing BlockVisibility's
    :param model_bag: Spectral line model bag
    :param continuum_model_bag: Continuum model bag
    :param context:
    :param kwargs: Parameters for functions in bags
    :return: bags of (deconvolved model, residual, restored)
    """
    assert isinstance(vis_bag, bag.Bag), vis_bag
    assert isinstance(model_bag, bag.Bag), model_bag
    
    if continuum_model_bag is not None:
        vis_bag = predict_bag(vis_bag, continuum_model_bag, **kwargs)
    
    return ical_pipeline_bag(vis_bag, model_bag, context=context, first_selfcal=numpy.infty, **kwargs)


def ical_pipeline_bag(vis_bag, model_bag, context='2d', first_selfcal=numpy.infty,
                      **kwargs) -> bag:
    """Create bag for ICAL pipeline
    
    :param vis_bag: Bag containing BlockVisibility's 
    :param model_bag: Image model in a bag
    :param context: Imaging context
    :param first_selfcal: First cycle for phase only selfcal
    :param kwargs: Parameters for functions in bags
    :return:
    """
    assert isinstance(vis_bag, bag.Bag), vis_bag
    assert isinstance(model_bag, bag.Bag), model_bag
    
    psf_bag = invert_bag(vis_bag, model_bag, context=context, dopsf=True, **kwargs)
    psf_bag = reify(psf_bag)
    
    # Make the predicted visibilities, selfcalibrate against it correcting the gains, then
    # form the residual visibility, then make the residual image
    model_vis_bag = reify(predict_bag(vis_bag, model_bag, context=context, **kwargs))
    if first_selfcal is not None and first_selfcal == 0:
        vis_bag = reify(calibrate_bag(vis_bag, model_vis_bag, **kwargs))
        res_vis_bag = reify(residual_vis_bag(vis_bag, model_vis_bag))
        res_bag = invert_bag(res_vis_bag, model_bag, context=context, dopsf=False, **kwargs)
    else:
        res_vis_bag = reify(residual_vis_bag(vis_bag, model_vis_bag))
        res_bag = invert_bag(res_vis_bag, model_bag, context=context, dopsf=False, **kwargs)

    res_bag = reify(res_bag)
    
    deconvolve_model_bag = reify(deconvolve_bag(res_bag, psf_bag, model_bag, **kwargs))
    
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):
            model_vis_bag = reify(predict_bag(vis_bag, deconvolve_model_bag, context=context, **kwargs))
            if first_selfcal is not None and cycle >= first_selfcal:
                vis_bag = reify(calibrate_bag(vis_bag, model_vis_bag, **kwargs))
                res_vis_bag = reify(residual_vis_bag(vis_bag, model_vis_bag))
                res_bag = invert_bag(res_vis_bag, model_bag, context=context, dopsf=False, **kwargs)
            else:
                res_vis_bag = reify(residual_vis_bag(vis_bag, model_vis_bag))
                res_bag = invert_bag(res_vis_bag, model_bag, context=context, dopsf=False, **kwargs)

            res_bag = reify(res_bag)
            deconvolve_model_bag = reify(deconvolve_bag(res_bag, psf_bag, deconvolve_model_bag,
                                                        **kwargs))
    
    res_bag = residual_image_bag(vis_bag, deconvolve_model_bag, context=context, **kwargs)
    res_bag = reify(res_bag)
    deconvolve_model_bag = reify(deconvolve_model_bag)
    rest_bag = restore_bag(deconvolve_model_bag, psf_bag, res_bag, **kwargs)
    rest_bag = reify(rest_bag)
    return bag.from_sequence([deconvolve_model_bag, res_bag, rest_bag])
