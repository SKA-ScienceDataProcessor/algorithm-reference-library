""" Pipeline functions. SDP standard pipelinee expressed as functions. This is quite slow and is provided mainly for
completeness. Use parallel versions pipelines/graphs.py for speed.

"""
import collections
import logging

from libs.data.data_models import BlockVisibility, GainTable
from libs.imaging import predict_skycomponent_visibility
from libs.visibility.base import copy_visibility

import numpy

from libs.calibration.calibration_control import calibrate_function, create_calibration_controls
from libs.data.data_models import Image, BlockVisibility, GainTable
from libs.data.parameters import get_parameter
from libs.image.deconvolution import deconvolve_cube, restore_cube
from libs.imaging import predict_skycomponent_visibility
from libs.imaging.imaging_functions import predict_function, invert_function
from libs.visibility.base import copy_visibility
from libs.visibility.coalesce import convert_blockvisibility_to_visibility

log = logging.getLogger(__name__)


def ical(block_vis: BlockVisibility, model: Image, components=None, context='2d', controls=None, **kwargs):
    """ Post observation image, deconvolve, and self-calibrate

    :param vis:
    :param model: Model image
    :param components: Initial components
    :param context: Imaging context
    :param controls: Calibration controls dictionary
    :return: model, residual, restored
    """
    nmajor = get_parameter(kwargs, 'nmajor', 5)
    log.info("ical: Performing %d major cycles" % nmajor)
    
    do_selfcal = get_parameter(kwargs, "do_selfcal", False)
    
    if controls is None:
        controls = create_calibration_controls(**kwargs)
    
    # The model is added to each major cycle and then the visibilities are
    # calculated from the full model
    vis = convert_blockvisibility_to_visibility(block_vis)
    block_vispred = copy_visibility(block_vis, zero=True)
    vispred = convert_blockvisibility_to_visibility(block_vispred)
    vispred.data['vis'][...] = 0.0
    visres = copy_visibility(vispred)
    
    vispred = predict_function(vispred, model, context=context, **kwargs)
    
    if components is not None:
        vispred = predict_skycomponent_visibility(vispred, components)
    
    if do_selfcal:
        vis, gaintables = calibrate_function(vis, vispred, 'TGB', controls, iteration=-1)
    
    visres.data['vis'] = vis.data['vis'] - vispred.data['vis']
    dirty, sumwt = invert_function(visres, model, context=context, **kwargs)
    log.info("Maximum in residual image is %.6f" % (numpy.max(numpy.abs(dirty.data))))
    
    psf, sumwt = invert_function(visres, model, dopsf=True, context=context, **kwargs)
    
    thresh = get_parameter(kwargs, "threshold", 0.0)
    
    for i in range(nmajor):
        log.info("ical: Start of major cycle %d of %d" % (i, nmajor))
        cc, res = deconvolve_cube(dirty, psf, **kwargs)
        model.data += cc.data
        vispred.data['vis'][...] = 0.0
        vispred = predict_function(vispred, model, context=context, **kwargs)
        if do_selfcal:
            vis, gaintables = calibrate_function(vis, vispred, 'TGB', controls, iteration=i)
        visres.data['vis'] = vis.data['vis'] - vispred.data['vis']
        
        dirty, sumwt = invert_function(visres, model, context=context, **kwargs)
        log.info("Maximum in residual image is %s" % (numpy.max(numpy.abs(dirty.data))))
        if numpy.abs(dirty.data).max() < 1.1 * thresh:
            log.info("ical: Reached stopping threshold %.6f Jy" % thresh)
            break
        log.info("ical: End of major cycle")
    
    log.info("ical: End of major cycles")
    restored = restore_cube(model, psf, dirty, **kwargs)
    
    return model, dirty, restored


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
    
    from libs.calibration.solvers import solve_gaintable
    for ichunk, vischunk in enumerate(vis):
        vispred = copy_visibility(vischunk, zero=True)
        vispred = predict_skycomponent_visibility(vispred, components)
        gt = solve_gaintable(vischunk, vispred, **kwargs)
        yield gt
