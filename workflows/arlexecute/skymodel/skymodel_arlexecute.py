import logging

import numpy

from data_models.memory_data_models import Image, GainTable, Visibility, BlockVisibility
from workflows.arlexecute.imaging.imaging_arlexecute import zero_list_arlexecute_workflow
from workflows.serial.imaging.imaging_serial import predict_list_serial_workflow, invert_list_serial_workflow
from wrappers.arlexecute.calibration.operations import apply_gaintable
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.imaging.base import predict_skycomponent_visibility
from wrappers.arlexecute.visibility.base import copy_visibility
from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility

log = logging.getLogger(__name__)


def predict_skymodel_list_arlexecute_workflow(vis_list, skymodel_list, context, vis_slices=1, facets=1,
                                              gcfcf=None, docal=False, **kwargs):
    """Predict from a skymodel, iterating over both the vis_list and skymodel
    
    The visibility and image are scattered, the visibility is predicted on each part, and then the
    parts are assembled.

    :param vis_list: List of Visibility data models
    :param skymodel_list: skymodel list
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param facets: Number of facets (per axis)
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param docal: Apply calibration table in skymodel
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists
   """
    
    assert len(vis_list) == len(skymodel_list)
    
    def dft_cal_sm(v, sm):
        assert isinstance(v, Visibility), v
        if len(sm.components) > 0:
            v = predict_skycomponent_visibility(v, sm.components)
            if docal and isinstance(sm.gaintable, GainTable):
                bv = convert_visibility_to_blockvisibility(v)
                bv = apply_gaintable(bv, sm.gaintable)
                v = convert_visibility_to_blockvisibility(bv)
        else:
            v.data['vis'][...] = 0.0 + 0.0j
        return v
    
    def fft_cal_sm(v, sm):
        assert isinstance(v, Visibility), v
        if isinstance(sm.image, Image) and isinstance(sm.gaintable, GainTable):
            if numpy.max(numpy.abs(sm.image.data)) > 0.0:
                v = predict_list_serial_workflow([v], [sm.image], context=context,
                                                 vis_slices=vis_slices, facets=facets, gcfcf=gcfcf,
                                                 **kwargs)[0]
                if docal:
                    bv = convert_visibility_to_blockvisibility(v)
                    bv = apply_gaintable(bv, sm.gaintable)
                    v = convert_blockvisibility_to_visibility(bv)
            else:
                v.data['vis'][...] = 0.0 + 0.0j
        
        else:
            v.data['vis'][...] = 0.0 + 0.0j
        return v
    
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


def invert_skymodel_list_arlexecute_workflow(vis_list, skymodel_list, context, vis_slices=1, facets=1,
                                             gcfcf=None, docal=False, **kwargs):
    """Calibrate and invert from a skymodel, iterating over the skymodel

    The visibility and image are scattered, the visibility is predicted and calibrated on each part, and then the
    parts are assembled.

    :param vis_list: List of Visibility data models
    :param skymodel_list: skymodel list
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param facets: Number of facets (per axis)
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param docal: Apply calibration table in skymodel
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists
   """
    
    assert len(vis_list) == len(skymodel_list)
    
    def ifft_ical_sm(v, sm):
        assert isinstance(v, Visibility), v
        assert isinstance(sm.image, Image), sm.image
        if docal and isinstance(sm.gaintable, GainTable):
            bv = convert_visibility_to_blockvisibility(v)
            bv = apply_gaintable(bv, sm.gaintable, inverse=True)
            v = convert_blockvisibility_to_visibility(bv)
        return invert_list_serial_workflow([v], [sm.image], context=context,
                                           vis_slices=vis_slices, facets=facets, gcfcf=gcfcf,
                                           **kwargs)[0]
    
    return [arlexecute.execute(ifft_ical_sm)(vis_list[0], sm) for sm in skymodel_list]
