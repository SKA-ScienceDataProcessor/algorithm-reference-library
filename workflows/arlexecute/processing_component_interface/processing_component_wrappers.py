""" Wrappers around ARL processing components, using JSON for configuration.

"""

import numpy

from data_models.buffer_data_models import BufferBlockVisibility, BufferImage, BufferSkyModel
from data_models.memory_data_models import SkyModel
from data_models.polarisation import PolarisationFrame
from libs.image.operations import create_image
from ..execution_support.arlexecute import arlexecute
from processing_components.image.gather_scatter import image_gather_channels
from processing_components.imaging.base import create_image_from_visibility, predict_skycomponent_visibility
from processing_components.skycomponent.operations import apply_beam_to_skycomponent
from processing_components.skycomponent.operations import insert_skycomponent
from processing_components.imaging.primary_beams import create_pb
from processing_components.simulation.testing_support import create_low_test_skycomponents_from_gleam

from workflows.arlexecute.simulation.simulation_arlexecute import simulate_arlexecute, corrupt_arlexecute
from workflows.arlexecute.imaging.imaging_arlexecute import predict_arlexecute
from workflows.arlexecute.imaging.imaging_arlexecute import remove_sumwt
from workflows.arlexecute.pipelines.pipeline_arlexecute import continuum_imaging_arlexecute, ical_arlexecute
from workflows.arlexecute.processing_component_interface.arl_json.json_helpers import json_to_quantity, \
    json_to_linspace, json_to_skycoord


def continuum_imaging_wrapper(conf):
    """Wrap continuum imaging pipeline
    
    :param conf:
    :return:
    """
    vis_list = BufferBlockVisibility(conf["buffer"], conf['inputs']['vis_list']).memory_data_model
    
    cellsize = json_to_quantity(conf["image"]["cellsize"]).to("rad").value
    npixel = conf["image"]["npixel"]
    pol_frame = PolarisationFrame(conf["image"]["polarisation_frame"])
    
    model_imagelist = [arlexecute.execute(create_image_from_visibility)(v, npixel=npixel, cellsize=cellsize,
                                                                        polarisation_frame=pol_frame)
                       for v in vis_list]
    
    future_vis_list = arlexecute.scatter(vis_list)
    
    result = continuum_imaging_arlexecute(vis_list=future_vis_list,
                                          model_imagelist=model_imagelist,
                                          context=conf["imaging"]["context"],
                                          scales=conf["deconvolution"]["scales"],
                                          algorithm=conf["deconvolution"]["algorithm"],
                                          nmoment=conf["deconvolution"]["nmoment"],
                                          niter=conf["deconvolution"]["niter"],
                                          fractional_threshold=conf["deconvolution"]["fractional_threshold"],
                                          threshold=conf["deconvolution"]["threshold"],
                                          nmajor=conf["deconvolution"]["nmajor"],
                                          gain=conf["deconvolution"]["gain"],
                                          deconvolve_facets=conf["deconvolution"]["deconvolve_facets"],
                                          deconvolve_overlap=conf["deconvolution"]["deconvolve_overlap"],
                                          deconvolve_taper=conf["deconvolution"]["deconvolve_taper"],
                                          vis_slices=conf["imaging"]["vis_slices"],
                                          psf_support=conf["deconvolution"]["psf_support"])

    def output_images(result):
        BufferSkyModel(conf["buffer"], conf['outputs']['skymodel'], SkyModel(images=result[0])).sync()
    
        deconvolved = image_gather_channels(result[0])
        residual = image_gather_channels(remove_sumwt(result[1]))
        restored = image_gather_channels(result[2])
    
        BufferImage(conf["buffer"], conf['outputs']['deconvolved'], deconvolved).sync()
        BufferImage(conf["buffer"], conf['outputs']['residual'], residual).sync()
        BufferImage(conf["buffer"], conf['outputs']['restored'], restored).sync()
    
        return result

    return arlexecute.execute(output_images)(result)


def ical_wrapper(conf):
    """ Wrap ICAL pipeline
    
    :param conf: Configuration from JSON file
    :return:
    """
    vis_list = BufferBlockVisibility(conf["buffer"], conf['inputs']['vis_list']).memory_data_model
    
    cellsize = json_to_quantity(conf["image"]["cellsize"]).to("rad").value
    npixel = conf["image"]["npixel"]
    pol_frame = PolarisationFrame(conf["image"]["polarisation_frame"])
    
    model_imagelist = [arlexecute.execute(create_image_from_visibility)(v, npixel=npixel, cellsize=cellsize,
                                                                        polarisation_frame=pol_frame)
                       for v in vis_list]
    
    future_vis_list = arlexecute.scatter(vis_list)
    
    result = ical_arlexecute(vis_list=future_vis_list,
                             model_imagelist=model_imagelist,
                             context=conf["imaging"]["context"],
                             scales=conf["deconvolution"]["scales"],
                             algorithm=conf["deconvolution"]["algorithm"],
                             nmoment=conf["deconvolution"]["nmoment"],
                             niter=conf["deconvolution"]["niter"],
                             fractional_threshold=conf["deconvolution"]["fractional_threshold"],
                             threshold=conf["deconvolution"]["threshold"],
                             nmajor=conf["deconvolution"]["nmajor"],
                             gain=conf["deconvolution"]["gain"],
                             deconvolve_facets=conf["deconvolution"]["deconvolve_facets"],
                             deconvolve_overlap=conf["deconvolution"]["deconvolve_overlap"],
                             deconvolve_taper=conf["deconvolution"]["deconvolve_taper"],
                             vis_slices=conf["imaging"]["vis_slices"],
                             psf_support=conf["deconvolution"]["psf_support"])
    
    def output_images(result):
        BufferSkyModel(conf["buffer"], conf['outputs']['skymodel'], SkyModel(images=result[0])).sync()
        
        deconvolved = image_gather_channels(result[0])
        residual = image_gather_channels(remove_sumwt(result[1]))
        restored = image_gather_channels(result[2])
        
        BufferImage(conf["buffer"], conf['outputs']['deconvolved'], deconvolved).sync()
        BufferImage(conf["buffer"], conf['outputs']['residual'], residual).sync()
        BufferImage(conf["buffer"], conf['outputs']['restored'], restored).sync()
        
        
        return result
    
    return arlexecute.execute(output_images)(result)


def create_vislist_wrapper(conf):
    """ Create an empty vislist
    
    :param conf: Configuration from JSON file
    :return:
    """
    configuration = conf['create_vislist']['configuration']
    rmax = conf['create_vislist']['rmax']
    phasecentre = json_to_skycoord(conf['create_vislist']['phasecentre'])
    frequency = json_to_linspace(conf['create_vislist']['frequency'])
    if conf['create_vislist']['frequency']['steps'] > 1:
        channel_bandwidth = numpy.array(conf['create_vislist']['frequency']['steps'] * [frequency[1] - frequency[0]])
    else:
        channel_bandwidth = numpy.array(conf['create_vislist']['frequency']['start'])
    
    times = json_to_linspace(conf['create_vislist']['times'])
    
    vis_list = simulate_arlexecute(configuration,
                                   rmax=rmax,
                                   frequency=frequency,
                                   channel_bandwidth=channel_bandwidth,
                                   times=times,
                                   phasecentre=phasecentre,
                                   order='frequency')
    
    def output_vislist(v):
        bdm = BufferBlockVisibility(conf["buffer"], conf["outputs"]["vis_list"], v)
        bdm.sync()
    
    return arlexecute.execute(output_vislist)(vis_list)


def create_skymodel_wrapper(conf):
    """ Wrapper to create skymodel
    
    :param conf: Configuration from JSON file
    :return:
    """
    
    cellsize = json_to_quantity(conf["image"]["cellsize"]).to("rad").value
    npixel = conf["image"]["npixel"]
    pol_frame = PolarisationFrame(conf["image"]["polarisation_frame"])
    phasecentre = json_to_skycoord(conf['image']['phasecentre'])
    frequency = json_to_linspace(conf['image']['frequency'])
    if conf['image']['frequency']['steps'] > 1:
        channel_bandwidth = numpy.array(conf['image']['frequency']['steps'] * [frequency[1] - frequency[0]])
    else:
        channel_bandwidth = numpy.array(conf['image']['frequency']['start'])
    
    flux_limit = json_to_quantity(conf['create_skymodel']['flux_limit']).to("Jy").value
    radius = json_to_quantity(conf['create_skymodel']['radius']).to('rad').value
    kind = conf['create_skymodel']['kind']
    
    models = [arlexecute.execute(create_image)(npixel=npixel, frequency=[frequency[f]],
                                               channel_bandwidth=[channel_bandwidth[f]],
                                               cellsize=cellsize,
                                               phasecentre=phasecentre,
                                               polarisation_frame=pol_frame)
              for f, freq in enumerate(frequency)]
    
    catalog = conf['create_skymodel']["catalog"]
    if catalog == "gleam":
        components = arlexecute.execute(create_low_test_skycomponents_from_gleam)(phasecentre=phasecentre,
                                                                                  polarisation_frame=pol_frame,
                                                                                  flux_limit=flux_limit,
                                                                                  frequency=frequency,
                                                                                  kind=kind,
                                                                                  radius=radius)
        if conf['create_skymodel']["fill_image"]:
            models = [arlexecute.execute(insert_skycomponent)(m, components) for m in models]
    
    elif catalog == "empty":
        components = []
    else:
        raise RuntimeError("Catalog %s is not supported" % catalog)
    
    def output_skymodel(model_list, comp_list):
        if conf['create_skymodel']["fill_image"]:
            skymodel = SkyModel(images=model_list, components=[])
        else:
            skymodel = SkyModel(images=[], components=comp_list)
        
        return BufferSkyModel(conf["buffer"], conf["outputs"]["skymodel"], skymodel).sync()
    
    return arlexecute.execute(output_skymodel)(models, components)


def predict_vislist_wrapper(conf):
    """Wrapper for prediction

    :param conf: Configuration from JSON file
    :return:
    """
    vis_list = BufferBlockVisibility(conf["buffer"], conf['inputs']['vis_list']).memory_data_model
    skymodel = BufferSkyModel(conf["buffer"], conf["inputs"]["skymodel"]).memory_data_model
    
    flux_limit = conf['primary_beam']['flux_limit']
    
    if conf["primary_beam"]["apply"]:
        def apply_pb_image(vt, model):
            telescope = vt.configuration.name
            pb = create_pb(model, telescope)
            model.data *= pb.data
            return model
        
        def apply_pb_comp(vt, model, comp):
            telescope = vt.configuration.name
            pb = create_pb(model, telescope)
            return apply_beam_to_skycomponent(comp, pb, flux_limit)
        
        image_list = [arlexecute.execute(apply_pb_image, nout=1)(v, skymodel.images[i]) for i, v in enumerate(vis_list)]
        if len(skymodel.components) > 1:
            component_list = [arlexecute.execute(apply_pb_comp, nout=1)(v, skymodel.images[i], skymodel.components)
                              for i, v in enumerate(vis_list)]
        else:
            component_list = []
    else:
        image_list = [skymodel.images[0] for v in vis_list]
        component_list = skymodel.components
    
    future_vis_list = arlexecute.scatter(vis_list)
    predicted_vis_list = [arlexecute.execute(predict_skycomponent_visibility)(v, component_list)
                          for v in future_vis_list]
    predicted_vis_list = predict_arlexecute(predicted_vis_list, image_list,
                                            context=conf['imaging']['context'],
                                            vis_slices=conf['imaging']['vis_slices'])
    
    def output_vislist(v):
        bdm = BufferBlockVisibility(conf["buffer"], conf["outputs"]["vis_list"], v)
        bdm.sync()
    
    return arlexecute.execute(output_vislist)(predicted_vis_list)


def corrupt_vislist_wrapper(conf):
    """Wrapper for corruption

    :param conf:
    :return:
    """
    vis_list = BufferBlockVisibility(conf["buffer"], conf['inputs']['vis_list']).memory_data_model
    phase_error = json_to_quantity(conf['corrupt_vislist']['phase_error']).to('rad').value
    
    corrupted_vislist = corrupt_arlexecute(vis_list,
                                           phase_error=phase_error,
                                           amplitude_error=conf['corrupt_vislist']['amplitude_error'])
    
    def output_vislist(v):
        bdm = BufferBlockVisibility(conf["buffer"], conf["outputs"]["vis_list"], v)
        bdm.sync()
    
    return arlexecute.execute(output_vislist)(corrupted_vislist)
