import logging

import numpy

from data_models.memory_data_models import Image, GainTable, Visibility
from processing_library.image.operations import copy_image
from workflows.serial.imaging.imaging_serial import predict_list_serial_workflow, invert_list_serial_workflow
from wrappers.arlexecute.calibration.operations import apply_gaintable
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.imaging.base import predict_skycomponent_visibility
from wrappers.arlexecute.skycomponent.base import copy_skycomponent
from wrappers.arlexecute.skycomponent.operations import apply_beam_to_skycomponent
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
    
    def ft_cal_sm(v, sm):
        assert isinstance(v, Visibility), v
        
        v.data['vis'][...] = 0.0 + 0.0j
        
        if len(sm.components) > 0:
            
            if isinstance(sm.mask, Image):
                comps = copy_skycomponent(sm.components)
                comps = apply_beam_to_skycomponent(comps, sm.mask)
                v = predict_skycomponent_visibility(v, comps)
            else:
                v = predict_skycomponent_visibility(v, sm.components)
        
        if isinstance(sm.image, Image):
            if numpy.max(numpy.abs(sm.image.data)) > 0.0:
                if isinstance(sm.mask, Image):
                    model = copy_image(sm.model)
                    model.data *= sm.mask.data
                else:
                    model = sm.model
                v = predict_list_serial_workflow([v], [model], context=context,
                                                 vis_slices=vis_slices, facets=facets, gcfcf=gcfcf,
                                                 **kwargs)[0]
        
        if docal and isinstance(sm.gaintable, GainTable):
            bv = convert_visibility_to_blockvisibility(v)
            bv = apply_gaintable(bv, sm.gaintable)
            v = convert_blockvisibility_to_visibility(bv)
        
        return v
    
    return [arlexecute.execute(ft_cal_sm, nout=1)(vis_list[i], skymodel_list[i])
            for i, _ in enumerate(vis_list)]


def invert_skymodel_list_arlexecute_workflow(vis_list, skymodel_list, context, vis_slices=1, facets=1,
                                             gcfcf=None, docal=False, **kwargs):
    """Calibrate and invert from a skymodel, iterating over the skymodel

    The visibility and image are scattered, the visibility is predicted and calibrated on each part, and then the
    parts are assembled. The mask if present, is multiplied in at the end.

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
    
    def ift_ical_sm(v, sm):
        assert isinstance(v, Visibility), v
        assert isinstance(sm.image, Image), sm.image
        
        if docal and isinstance(sm.gaintable, GainTable):
            bv = convert_visibility_to_blockvisibility(v)
            bv = apply_gaintable(bv, sm.gaintable, inverse=True)
            v = convert_blockvisibility_to_visibility(bv)
            
        result = invert_list_serial_workflow([v], [sm.image], context=context,
                                             vis_slices=vis_slices, facets=facets, gcfcf=gcfcf,
                                             **kwargs)[0]
        if isinstance(sm.mask, Image):
            result[0].data *= sm.mask
        return result
    
    return [arlexecute.execute(ift_ical_sm)(vis_list[0], sm) for sm in skymodel_list]
