""" Pipeline functions. SDP standard pipelinee expressed as functions. This is quite slow and is provided mainly for
completeness. Use parallel versions pipelines/components.py for speed.

"""
from data_models.parameters import get_parameter

from ..calibration.calibration_serial import calibrate_list_serial_workflow
from ..imaging.imaging_serial import invert_list_serial_workflow, residual_list_serial_workflow, \
    predict_list_serial_workflow, zero_list_serial_workflow, subtract_list_serial_workflow, \
    restore_list_serial_workflow, \
    deconvolve_list_serial_workflow


def ical_list_serial_workflow(vis_list, model_imagelist, context='2d', calibration_context='TG', do_selfcal=True,
                                  **kwargs):
    """Create graph for ICAL pipeline

    :param vis_list:
    :param model_imagelist:
    :param context: imaging context e.g. '2d'
    :param calibration_context: Sequence of calibration steps e.g. TGB
    :param do_selfcal: Do the selfcalibration?
    :param kwargs: Parameters for functions in components
    :return:
    """
    psf_imagelist = invert_list_serial_workflow(vis_list, model_imagelist, dopsf=True, context=context, **kwargs)
    
    model_vislist = zero_list_serial_workflow(vis_list)
    model_vislist = predict_list_serial_workflow(model_vislist, model_imagelist, context=context, **kwargs)
    if do_selfcal:
        # Make the predicted visibilities, selfcalibrate against it correcting the gains, then
        # form the residual visibility, then make the residual image
        vis_list = calibrate_list_serial_workflow(vis_list, model_vislist,
                                                      calibration_context=calibration_context, **kwargs)
        residual_vislist = subtract_list_serial_workflow(vis_list, model_vislist)
        residual_imagelist = invert_list_serial_workflow(residual_vislist, model_imagelist, dopsf=True,
                                                             context=context,
                                                             iteration=0, **kwargs)
    else:
        # If we are not selfcalibrating it's much easier and we can avoid an unnecessary round of gather/scatter
        # for visibility partitioning such as timeslices and wstack.
        residual_imagelist = residual_list_serial_workflow(vis_list, model_imagelist, context=context, **kwargs)
    
    deconvolve_model_imagelist, _ = deconvolve_list_serial_workflow(residual_imagelist, psf_imagelist,
                                                                        model_imagelist,
                                                                        prefix='cycle 0', **kwargs)
    
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):
            if do_selfcal:
                model_vislist = zero_list_serial_workflow(vis_list)
                model_vislist = predict_list_serial_workflow(model_vislist, deconvolve_model_imagelist,
                                                                 context=context, **kwargs)
                vis_list = calibrate_list_serial_workflow(vis_list, model_vislist,
                                                              calibration_context=calibration_context,
                                                              iteration=cycle, **kwargs)
                residual_vislist = subtract_list_serial_workflow(vis_list, model_vislist)
                residual_imagelist = invert_list_serial_workflow(residual_vislist, model_imagelist, dopsf=False,
                                                                     context=context, **kwargs)
            else:
                residual_imagelist = residual_list_serial_workflow(vis_list, deconvolve_model_imagelist,
                                                                       context=context, **kwargs)
            
            prefix = "cycle %d" % (cycle + 1)
            deconvolve_model_imagelist, _ = deconvolve_list_serial_workflow(residual_imagelist, psf_imagelist,
                                                                                deconvolve_model_imagelist,
                                                                                prefix=prefix,
                                                                                **kwargs)
    residual_imagelist = residual_list_serial_workflow(vis_list, deconvolve_model_imagelist, context=context,
                                                           **kwargs)
    restore_imagelist = restore_list_serial_workflow(deconvolve_model_imagelist, psf_imagelist, residual_imagelist)
    
    return (deconvolve_model_imagelist, residual_imagelist, restore_imagelist)


def continuum_imaging_list_serial_workflow(vis_list, model_imagelist, context='2d', **kwargs):
    """ Create graph for the continuum imaging pipeline.

    Same as ICAL but with no selfcal.

    :param vis_list:
    :param model_imagelist:
    :param context: Imaging context
    :param kwargs: Parameters for functions in components
    :return:
    """
    psf_imagelist = invert_list_serial_workflow(vis_list, model_imagelist, dopsf=True, context=context, **kwargs)
    
    residual_imagelist = residual_list_serial_workflow(vis_list, model_imagelist, context=context, **kwargs)
    deconvolve_model_imagelist, _ = deconvolve_list_serial_workflow(residual_imagelist, psf_imagelist,
                                                                        model_imagelist,
                                                                        prefix='cycle 0',
                                                                        **kwargs)
    
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):
            prefix = "cycle %d" % (cycle + 1)
            residual_imagelist = residual_list_serial_workflow(vis_list, deconvolve_model_imagelist,
                                                                   context=context, **kwargs)
            deconvolve_model_imagelist, _ = deconvolve_list_serial_workflow(residual_imagelist, psf_imagelist,
                                                                                deconvolve_model_imagelist,
                                                                                prefix=prefix,
                                                                                **kwargs)
    
    residual_imagelist = residual_list_serial_workflow(vis_list, deconvolve_model_imagelist, context=context,
                                                           **kwargs)
    restore_imagelist = restore_list_serial_workflow(deconvolve_model_imagelist, psf_imagelist, residual_imagelist)
    return (deconvolve_model_imagelist, residual_imagelist, restore_imagelist)


def spectral_line_imaging_list_serial_workflow(vis_list, model_imagelist, continuum_model_imagelist=None,
                                                   context='2d', **kwargs):
    """Create graph for spectral line imaging pipeline

    Uses the continuum imaging serial pipeline after subtraction of a continuum model

    :param vis_list: List of visibility components
    :param model_imagelist: Spectral line model graph
    :param continuum_model_imagelist: Continuum model list
    :param context: Imaging context
    :param kwargs: Parameters for functions in components
    :return: (deconvolved model, residual, restored)
    """
    if continuum_model_imagelist is not None:
        vis_list = predict_list_serial_workflow(vis_list, continuum_model_imagelist, context=context, **kwargs)
    
    return continuum_imaging_list_serial_workflow(vis_list, model_imagelist, context=context, **kwargs)

