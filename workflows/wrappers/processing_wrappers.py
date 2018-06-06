from data_models.data_model_helpers import import_blockvisibility_from_hdf5
from data_models.parameters import arl_path
from data_models.polarisation import PolarisationFrame
from processing_components.component_support.arlexecute import arlexecute
from processing_components.image.operations import export_image_to_fits
from processing_components.imaging.base import create_image_from_visibility
from processing_components.pipelines.pipeline_components import continuum_imaging_component


def initialise_vislist_and_model_wrapper(conf):
    """
    
    :param conf:
    :return:
    """


def continuum_imaging_component_wrapper(conf):
    vis_list = import_blockvisibility_from_hdf5(arl_path(conf["inputs"]["vis_list"]))
    
    import astropy.units as u
    cellsize = conf["image"]["cellsize"]["value"] * (u.Unit(conf["image"]["cellsize"]["unit"]).to("rad"))
    npixel = conf["image"]["npixel"]
    pol_frame = PolarisationFrame(conf["image"]["polarisation_frame"])
    
    model_imagelist = [arlexecute.execute(create_image_from_visibility)(v, npixel=npixel, cellsize=cellsize,
                                                                        polarisation_frame=pol_frame)
                       for v in vis_list]
    
    result = continuum_imaging_component(vis_list=vis_list,
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


def create_model_from_vislist_wrapper(conf):
    vis_list = conf["inputs"]["vis_list"]
    if isinstance(vis_list, str):
        vis_list = import_blockvisibility_from_hdf5(arl_path(conf["inputs"]["vis_list"]))
    
    import astropy.units as u
    cellsize = conf["image"]["cellsize"]["value"] * (u.Unit(conf["image"]["cellsize"]["unit"]).to("rad"))
    npixel = conf["image"]["npixel"]
    pol_frame = PolarisationFrame(conf["image"]["polarisation_frame"])
    
    return [arlexecute.execute(create_image_from_visibility)(v, npixel=npixel, cellsize=cellsize,
                                                             polarisation_frame=pol_frame)
            for v in vis_list]
