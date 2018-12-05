import logging

from workflows.arlexecute.imaging.imaging_arlexecute import zero_list_arlexecute_workflow, \
    predict_list_arlexecute_workflow
from workflows.serial.imaging.imaging_serial import predict_list_serial_workflow
from wrappers.arlexecute.calibration.operations import apply_gaintable
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.imaging.base import predict_skycomponent_visibility
from wrappers.arlexecute.visibility.base import copy_visibility
from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility

log = logging.getLogger(__name__)


def predict_skymodel_list_arlexecute_workflow(vis_list, skymodel_list, context, vis_slices=1, facets=1,
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
        return sm.images[0]
    
    comp = [arlexecute.execute(extract_comps, nout=1)(sm) for sm in skymodel_list]
    images = [arlexecute.execute(extract_image, nout=1)(sm) for sm in skymodel_list]
    
    dft_vis_list = zero_list_arlexecute_workflow(vis_list)
    dft_vis_list = [arlexecute.execute(predict_skycomponent_visibility, nout=1)(dft_vis_list[i], comp[i])
                    for i, _ in enumerate(dft_vis_list)]
    
    fft_vis_list = zero_list_arlexecute_workflow(vis_list)
    fft_vis_list = predict_list_arlexecute_workflow(fft_vis_list, images, context=context,
                                                    vis_slices=vis_slices, facets=facets, gcfcf=gcfcf, **kwargs)
    
    def vis_add(v1, v2):
        vout = copy_visibility(v1)
        vout.data['vis'] += v2.data['vis']
        return vout
    
    return [arlexecute.execute(vis_add, nout=1)(dft_vis_list[i], fft_vis_list[i])
            for i, _ in enumerate(dft_vis_list)]


def predictcal_skymodel_list_arlexecute_workflow(vis_list, skymodel_list, context, vis_slices=1, facets=1,
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
            bv = apply_gaintable(bv, sm.gaintables[0])
        return bv
    
    def fft_cal_sm(bv, sm):
        if len(sm.images) > 0:
            v = convert_blockvisibility_to_visibility(bv)
            v = predict_list_serial_workflow([v], sm.images, context=context,
                                            vis_slices=vis_slices, facets=facets, gcfcf=gcfcf,
                                            **kwargs)[0]
            bv = convert_visibility_to_blockvisibility(v)
            bv = apply_gaintable(bv, sm.gaintables[0])
        return bv
    
    dft_vis_list = zero_list_arlexecute_workflow(vis_list)
    dft_vis_list = [arlexecute.execute(dft_cal_sm, nout=1)(dft_vis_list[i], skymodel_list[i])
                    for i, _ in enumerate(dft_vis_list)]
    
    fft_vis_list = zero_list_arlexecute_workflow(vis_list)
    fft_vis_list = [arlexecute.execute(fft_cal_sm)(fft_vis_list[i], skymodel_list[i])
                    for i, _ in enumerate(fft_vis_list)]
    
    def vis_add(v1, v2):
        vout = copy_visibility(v1)
        vout.data['vis'] += v2.data['vis']
        return vout
    
    return [arlexecute.execute(vis_add, nout=1)(dft_vis_list[i], fft_vis_list[i])
            for i, _ in enumerate(dft_vis_list)]
