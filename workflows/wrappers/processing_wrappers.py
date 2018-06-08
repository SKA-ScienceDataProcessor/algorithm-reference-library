""" Wrappers around ARL processing components, using JSON for configuration

"""

import numpy

from data_models.data_model_helpers import export_blockvisibility_to_hdf5, export_skymodel_to_hdf5
from data_models.data_model_helpers import import_blockvisibility_from_hdf5, import_skymodel_from_hdf5
from data_models.memory_data_models import SkyModel
from data_models.parameters import arl_path
from data_models.polarisation import PolarisationFrame

from processing_components.component_support.arlexecute import arlexecute
from processing_components.image.operations import export_image_to_fits
from processing_components.imaging.base import create_image_from_visibility
from processing_components.pipelines.pipeline_components import continuum_imaging_component
from processing_components.util.support_components import simulate_component, corrupt_component
from processing_components.util.testing_support import create_low_test_image_from_gleam
from processing_components.imaging.imaging_components import predict_component

from workflows.wrappers.arl_json.json_helpers import json_to_quantity, json_to_linspace, json_to_skycoord


def continuum_imaging_wrapper(conf):
    vis_list = import_blockvisibility_from_hdf5(arl_path(conf["inputs"]["vis_list"]))
    
    cellsize = json_to_quantity(conf["image"]["cellsize"]).to("rad").value
    npixel = conf["image"]["npixel"]
    pol_frame = PolarisationFrame(conf["image"]["polarisation_frame"])
    
    model_imagelist = [arlexecute.execute(create_image_from_visibility)(v, npixel=npixel, cellsize=cellsize,
                                                                        polarisation_frame=pol_frame)
                       for v in vis_list]
    
    future_vis_list = arlexecute.scatter(vis_list)
    
    result = continuum_imaging_component(vis_list=future_vis_list,
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
        deconvolved = result[0][0]
        residual = result[1][0]
        restored = result[2][0]

        export_image_to_fits(deconvolved, conf['outputs']['deconvolved'])
        export_image_to_fits(restored, conf['outputs']['restored'])
        export_image_to_fits(residual, conf['outputs']['residual'])
        
        return result
    
    return arlexecute.execute(output_images)(result)


def create_vislist_wrapper(conf):
    configuration = conf['simulate']['configuration']
    rmax = conf['simulate']['rmax']
    phasecentre = json_to_skycoord(conf['simulate']['phasecentre'])
    frequency = json_to_linspace(conf['simulate']['frequency'])
    if conf['simulate']['frequency']['steps'] > 1:
        channel_bandwidth = numpy.array(conf['simulate']['frequency']['steps'] * [frequency[1] - frequency[0]])
    else:
        channel_bandwidth = numpy.array(conf['simulate']['frequency']['start'])
    
    times = json_to_linspace(conf['simulate']['times'])
    
    vis_list = simulate_component(configuration,
                                  rmax=rmax,
                                  frequency=frequency,
                                  channel_bandwidth=channel_bandwidth,
                                  times=times,
                                  phasecentre=phasecentre,
                                  order='frequency')
    
    def output_vis_list(v):
        export_blockvisibility_to_hdf5(v, conf["outputs"]["vis_list"])
        
        return True
    
    return arlexecute.execute(output_vis_list)(vis_list)


def create_model_from_vislist_wrapper(conf):
    vis_list = import_blockvisibility_from_hdf5(arl_path(conf["inputs"]["vis_list"]))
    
    cellsize = json_to_quantity(conf["image"]["cellsize"]["value"]).to("rad")
    npixel = conf["image"]["npixel"]
    pol_frame = PolarisationFrame(conf["image"]["polarisation_frame"])
    
    return [arlexecute.execute(create_image_from_visibility)(v, npixel=npixel, cellsize=cellsize,
                                                             polarisation_frame=pol_frame)
            for v in vis_list]


def create_skymodel_wrapper(conf):
    """ Wrapper to create skymodel
    
    :param conf:
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
    
    catalog = conf['create_skymodel']["catalog"]
    if catalog == "gleam":
        model = [arlexecute.execute(create_low_test_image_from_gleam)(npixel=npixel,
                                                                      frequency=[frequency[f]],
                                                                      channel_bandwidth=[channel_bandwidth[f]],
                                                                      cellsize=cellsize,
                                                                      phasecentre=phasecentre,
                                                                      polarisation_frame=pol_frame,
                                                                      flux_limit=flux_limit,
                                                                      radius=radius)
                 for f, freq in enumerate(frequency)]
    else:
        raise RuntimeError("Catalog %s is not supported" % catalog)
    
    def output_skymodel(m):
        skymodel = SkyModel(images=m)
        export_skymodel_to_hdf5(skymodel, arl_path(conf["outputs"]["skymodel"]))
        return True
    
    return arlexecute.execute(output_skymodel)(model)

def simulate_wrapper(conf):
    """Wrapper for simulation i.e. prediction and corruption
    
    :param conf:
    :return:
    """
    vis_list = import_blockvisibility_from_hdf5(conf['inputs']['vis_list'])
    skymodel = import_skymodel_from_hdf5(conf['inputs']['skymodel'])

    predicted_vislist = predict_component(vis_list, skymodel.images,
                                          context=conf['imaging']['context'],
                                          vis_slices=conf['imaging']['vis_slices'])
    
    phase_error = json_to_quantity(conf['simulate']['phase_error']).to('rad').value
    
    corrupted_vislist = corrupt_component(predicted_vislist,
                                          phase_error=phase_error,
                                          amplitude_error=conf['simulate']['amplitude_error'])

    def output_vislist(v):
        return export_blockvisibility_to_hdf5(corrupted_vislist, conf['outputs']['vis_list'])
    
    return arlexecute.execute(output_vislist)(vis_list)
