import logging

from workflows.serial.imaging.imaging_serial import zero_list_serial_workflow, \
    predict_list_serial_workflow
from wrappers.serial.imaging.base import predict_skycomponent_visibility
from wrappers.serial.visibility.base import copy_visibility

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
        return sm.images[0]
    
    comp = [extract_comps(sm) for sm in skymodel_list]
    images =[extract_image(sm) for sm in skymodel_list]
    
    dft_vis_list = zero_list_serial_workflow(vis_list)
    dft_vis_list =[predict_skycomponent_visibility(dft_vis_list[i], comp[i]) for i, _ in enumerate(dft_vis_list)]
    
    fft_vis_list = zero_list_serial_workflow(vis_list)
    fft_vis_list = predict_list_serial_workflow(fft_vis_list, images, context=context,
                                                vis_slices=vis_slices, facets=facets, gcfcf=gcfcf, **kwargs)
    
    def vis_add(v1, v2):
        vout = copy_visibility(v1)
        vout.data['vis'] += v2.data['vis']
        return vout
    
    return [vis_add(dft_vis_list[i], fft_vis_list[i]) for i, _ in enumerate(dft_vis_list)]
