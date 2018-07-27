""" Pipeline functions. SDP standard pipelinee expressed as functions. This is quite slow and is provided mainly for
completeness. Use parallel versions pipelines/components.py for speed.

"""
import collections
import logging

import numpy

from data_models.memory_data_models import Image, BlockVisibility, GainTable
from data_models.parameters import get_parameter

from processing_components.calibration.calibration import solve_gaintable
from processing_components.calibration.calibration_control import calibrate_function, create_calibration_controls
from processing_components.image.deconvolution import deconvolve_cube, restore_cube
from processing_components.imaging.base import predict_skycomponent_visibility
from workflows.serial.imaging.imaging_serial import predict_serial, invert_serial
from processing_components.visibility.base import copy_visibility
from processing_components.visibility.coalesce import convert_blockvisibility_to_visibility

log = logging.getLogger(__name__)


def ical_serial(block_vis: BlockVisibility, model: Image, components=None, context='2d', controls=None, **kwargs):
    """ Post observation image, deconvolve, and self-calibrate

    :param vis:
    :param model: Model image
    :param components: Initial components
    :param context: Imaging context
    :param controls: calibration controls dictionary
    :return: model, residual, restored
    """
    nmajor = get_parameter(kwargs, 'nmajor', 5)
    log.info("ical_serial: Performing %d major cycles" % nmajor)
    
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
    
    vispred = predict_serial(vispred, model, context=context, **kwargs)
    
    if components is not None:
        vispred = predict_skycomponent_visibility(vispred, components)
    
    if do_selfcal:
        vis, gaintables = calibrate_function(vis, vispred, 'TGB', controls, iteration=-1)
    
    visres.data['vis'] = vis.data['vis'] - vispred.data['vis']
    dirty, sumwt = invert_serial(visres, model, context=context, **kwargs)
    log.info("Maximum in residual image is %.6f" % (numpy.max(numpy.abs(dirty.data))))
    
    psf, sumwt = invert_serial(visres, model, dopsf=True, context=context, **kwargs)
    
    thresh = get_parameter(kwargs, "threshold", 0.0)
    
    for i in range(nmajor):
        log.info("ical_serial: Start of major cycle %d of %d" % (i, nmajor))
        cc, res = deconvolve_cube(dirty, psf, **kwargs)
        model.data += cc.data
        vispred.data['vis'][...] = 0.0
        vispred = predict_serial(vispred, model, context=context, **kwargs)
        if do_selfcal:
            vis, gaintables = calibrate_function(vis, vispred, 'TGB', controls, iteration=i)
        visres.data['vis'] = vis.data['vis'] - vispred.data['vis']
        
        dirty, sumwt = invert_serial(visres, model, context=context, **kwargs)
        log.info("Maximum in residual image is %s" % (numpy.max(numpy.abs(dirty.data))))
        if numpy.abs(dirty.data).max() < 1.1 * thresh:
            log.info("ical_serial: Reached stopping threshold %.6f Jy" % thresh)
            break
        log.info("ical_serial: End of major cycle")
    
    log.info("ical_serial: End of major cycles")
    restored = restore_cube(model, psf, dirty, **kwargs)
    
    return model, dirty, restored


def rcal_serial(vis: BlockVisibility, components, **kwargs) -> GainTable:
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
        vispred = predict_skycomponent_visibility(vispred, components)
        gt = solve_gaintable(vischunk, vispred, **kwargs)
        yield gt
