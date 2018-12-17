import logging

from workflows.serial.imaging.imaging_serial import zero_list_serial_workflow, \
    predict_list_serial_workflow, invert_list_serial_workflow
from wrappers.serial.calibration.operations import apply_gaintable
from wrappers.serial.imaging.base import predict_skycomponent_visibility
from wrappers.serial.visibility.base import copy_visibility
from wrappers.serial.visibility.coalesce import convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility

log = logging.getLogger(__name__)


def predict_skymodel_list_serial_workflow(vis_list, skymodel_list, context, vis_slices=1, facets=1,
                                          gcfcf=None, **kwargs):
    """Predict from a skymodel, iterating over both the vis_list and skymodel
    
    The visibility and image are scattered, the visibility is predicted on each part, and then the
    parts are assembled.

    :param vis_list:
    :param skymodel_list: skymodel list
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param facets: Number of facets (per axis)
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists
   """
    
    def extract_comps(sm):
        return sm.components
    
    def extract_image(sm):
        return sm.image
    
    comp = [extract_comps(sm) for sm in skymodel_list]
    images = [extract_image(sm) for sm in skymodel_list]
    
    dft_vis_list = zero_list_serial_workflow(vis_list)
    dft_vis_list = [predict_skycomponent_visibility(dft_vis_list[i], comp[i]) for i, _ in enumerate(dft_vis_list)]
    
    fft_vis_list = zero_list_serial_workflow(vis_list)
    fft_vis_list = [predict_list_serial_workflow([fft_vis_list[i]], images[i], context=context,
                                                vis_slices=vis_slices, facets=facets,
                                                gcfcf=gcfcf, **kwargs)
    for i, _ in enumerate(fft_vis_list)]
    
    def vis_add(v1, v2):
        vout = copy_visibility(v1)
        vout.data['vis'] += v2.data['vis']
        return vout
    
    return [vis_add(dft_vis_list[i], fft_vis_list[i]) for i, _ in enumerate(dft_vis_list)]


def predictcal_skymodel_list_serial_workflow(vis_list, skymodel_list, context, vis_slices=1, facets=1,
                                             gcfcf=None, **kwargs):
    """Predict and calibrate from a skymodel, iterating over both the vis_list and skymodel

    The visibility and image are scattered, the visibility is predicted and calibrated on each part, and then the
    parts are assembled.

    :param vis_list:
    :param skymodel_list: skymodel list
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param facets: Number of facets (per axis)
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists
   """
    
    assert len(vis_list) == len(skymodel_list)
    
    def dft_cal_sm(bv, sm):
        if len(sm.components) > 0:
            bv = predict_skycomponent_visibility(bv, sm.components)
            bv = apply_gaintable(bv, sm.gaintable)
        return bv
    
    def fft_cal_sm(bv, sm):
        if sm.image is not None:
            v = convert_blockvisibility_to_visibility(bv)
            v = predict_list_serial_workflow([v], [sm.image], context=context,
                                             vis_slices=vis_slices, facets=facets, gcfcf=gcfcf,
                                             **kwargs)[0]
            bv = convert_visibility_to_blockvisibility(v)
            bv = apply_gaintable(bv, sm.gaintable)
        return bv
    
    dft_vis_list = zero_list_serial_workflow(vis_list)
    dft_vis_list = [dft_cal_sm(dft_vis_list[i], skymodel_list[i])
                    for i, _ in enumerate(dft_vis_list)]
    
    fft_vis_list = zero_list_serial_workflow(vis_list)
    fft_vis_list = [fft_cal_sm(fft_vis_list[i], skymodel_list[i])
                    for i, _ in enumerate(fft_vis_list)]
    
    def vis_add(v1, v2):
        vout = copy_visibility(v1)
        vout.data['vis'] += v2.data['vis']
        return vout
    
    return [vis_add(dft_vis_list[i], fft_vis_list[i])
            for i, _ in enumerate(dft_vis_list)]


def invertcal_skymodel_list_serial_workflow(vis_list, skymodel_list, context, vis_slices=1, facets=1,
                                            gcfcf=None, **kwargs):
    """Calibrate and invert from a skymodel, iterating over both the vis_list and skymodel

    The visibility and image are scattered, the visibility is predicted and calibrated on each part, and then the
    parts are assembled.

    :param vis_list:
    :param skymodel_list: skymodel list
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param facets: Number of facets (per axis)
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists
   """
    
    assert len(vis_list) == len(skymodel_list)
    
    bv = vis_list[0]
    results = list()
    for sm in skymodel_list:
        if sm.image is not None:
            cbv = copy_visibility(bv)
            cbv = apply_gaintable(cbv, sm.gaintable, inverse=True)
            cv = convert_blockvisibility_to_visibility(cbv)
            results.append(invert_list_serial_workflow([cv], [sm.image], context=context,
                                                       vis_slices=vis_slices, facets=facets, gcfcf=gcfcf,
                                                       **kwargs)[0])
        else:
            results.append(None)
    
    return results
