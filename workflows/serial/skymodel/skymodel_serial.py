import logging

import numpy

from data_models.memory_data_models import Image, GainTable, Visibility, SkyModel, ConvolutionFunction
from processing_library.image.operations import copy_image
from workflows.serial.imaging.imaging_serial import predict_list_serial_workflow, invert_list_serial_workflow
from wrappers.serial.calibration.operations import apply_gaintable
from wrappers.serial.imaging.base import predict_skycomponent_visibility
from wrappers.serial.skycomponent.base import copy_skycomponent
from wrappers.serial.skycomponent.operations import apply_beam_to_skycomponent
from wrappers.serial.visibility.base import copy_visibility
from wrappers.serial.visibility.coalesce import convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility

log = logging.getLogger(__name__)


def predict_skymodel_list_serial_workflow(obsvis, skymodel_list, context, vis_slices=1, facets=1,
                                          gcfcf=None, docal=False, **kwargs):
    """Predict from a list of skymodels, producing one visibility per skymodel

    :param obsvis: "Observed Visibility"
    :param skymodel_list: skymodel list
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param facets: Number of facets (per axis)
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param docal: Apply calibration table in skymodel
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists
   """
    
    def ft_cal_sm(ov, sm, g):
        assert isinstance(ov, Visibility), ov
        assert isinstance(sm, SkyModel), sm
        if g is not None:
            assert len(g) == 2, g
            assert isinstance(g[0], Image), g[0]
            assert isinstance(g[1], ConvolutionFunction), g[1]

        v = copy_visibility(ov)
        
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
                    model = copy_image(sm.image)
                    model.data *= sm.mask.data
                else:
                    model = sm.image
                v = predict_list_serial_workflow([v], [model], context=context,
                                                 vis_slices=vis_slices, facets=facets, gcfcf=[g],
                                                 **kwargs)[0]
        
        if docal and isinstance(sm.gaintable, GainTable):
            bv = convert_visibility_to_blockvisibility(v)
            bv = apply_gaintable(bv, sm.gaintable, inverse=True)
            v = convert_blockvisibility_to_visibility(bv)
        
        return v
    
    if gcfcf is None:
       return [ft_cal_sm(obsvis, sm, None)
                for ism, sm in enumerate(skymodel_list)]
    else:
        return [ft_cal_sm(obsvis, sm, gcfcf[ism])
                for ism, sm in enumerate(skymodel_list)]


def invert_skymodel_list_serial_workflow(vis_list, skymodel_list, context, vis_slices=1, facets=1,
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
    :return: List of (image, weight) tuples)
   """
    
    def ift_ical_sm(v, sm, g):
        assert isinstance(v, Visibility), v
        assert isinstance(sm, SkyModel), sm
        if g is not None:
            assert len(g) == 2, g
            assert isinstance(g[0], Image), g[0]
            assert isinstance(g[1], ConvolutionFunction), g[1]
        
        if docal and isinstance(sm.gaintable, GainTable):
            bv = convert_visibility_to_blockvisibility(v)
            bv = apply_gaintable(bv, sm.gaintable)
            v = convert_blockvisibility_to_visibility(bv)
        
        result = invert_list_serial_workflow([v], [sm.image], context=context,
                                             vis_slices=vis_slices, facets=facets, gcfcf=[g],
                                             **kwargs)[0]
        if isinstance(sm.mask, Image):
            result[0].data *= sm.mask.data
        
        return result
    if gcfcf is None:
        return [ift_ical_sm(vis_list[i], sm, None)
                for i, sm in enumerate(skymodel_list)]
    else:
        return [ift_ical_sm(vis_list[i], sm, gcfcf[i])
                for i, sm in enumerate(skymodel_list)]
    


def crosssubtract_datamodels_skymodel_list_serial_workflow(obsvis, modelvis_list):
    """Form data models by subtracting sum from the observed and adding back each model in turn

    vmodel[p] = vobs - sum(i!=p) modelvis[i]

    This is the E step in the Expectation-Maximisation algorithm.

    :param obsvis: "Observed" visibility
    :param modelvis_list: List of Visibility data model predictions
    :return: List of (image, weight) tuples)
   """
    
    # Now do the meaty part. We probably want to refactor this for performance once it works.
    def vsum(ov, mv):
        # Observed vis minus the sum of all predictions
        verr = copy_visibility(ov)
        for m in mv:
            verr.data['vis'] -= m.data['vis']
        # Now add back each model in turn
        result = list()
        for m in mv:
            vr = copy_visibility(verr)
            vr.data['vis'] += m.data['vis']
            result.append(vr)
        assert len(result) == len(mv)
        return result
    
    return vsum(obsvis, modelvis_list)


def convolve_skymodel_list_serial_workflow(obsvis, skymodel_list, context, vis_slices=1, facets=1,
                                               gcfcf=None, **kwargs):
    """Form residual image from observed visibility and a set of skymodel without calibration

    This is similar to convolving the skymodel images with the PSF

    :param vis_list: List of Visibility data models
    :param skymodel_list: skymodel list
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param facets: Number of facets (per axis)
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param docal: Apply calibration table in skymodel
    :param kwargs: Parameters for functions in components
    :return: List of (image, weight) tuples)
   """
    
    def ft_ift_sm(ov, sm, g):
        assert isinstance(ov, Visibility), ov
        assert isinstance(sm, SkyModel), sm
        if g is not None:
            assert len(g) == 2, g
            assert isinstance(g[0], Image), g[0]
            assert isinstance(g[1], ConvolutionFunction), g[1]
            
        v = copy_visibility(ov)
        
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
                    model = copy_image(sm.image)
                    model.data *= sm.mask.data
                else:
                    model = sm.image
                v = predict_list_serial_workflow([v], [model], context=context,
                                                 vis_slices=vis_slices, facets=facets, gcfcf=[g],
                                                 **kwargs)[0]
        
        assert isinstance(sm.image, Image), sm.image
        
        result = invert_list_serial_workflow([v], [sm.image], context=context,
                                             vis_slices=vis_slices, facets=facets, gcfcf=gcfcf,
                                             **kwargs)[0]
        if isinstance(sm.mask, Image):
            result[0].data *= sm.mask.data
        return result
    
    if gcfcf is None:
        return [ft_ift_sm(obsvis, sm, None)
                for ism, sm in enumerate(skymodel_list)]
    else:
        return [ft_ift_sm(obsvis, sm, gcfcf[ism])
                for ism, sm in enumerate(skymodel_list)]
