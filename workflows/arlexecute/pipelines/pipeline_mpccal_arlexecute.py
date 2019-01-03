""" Pipeline functions. SDP standard pipelines expressed as functions.
"""

from data_models.parameters import get_parameter
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.griddata.kernels import create_pswf_convolutionfunction
from wrappers.arlexecute.visibility.base import copy_visibility
from ..calibration.calibration_arlexecute import calibrate_list_arlexecute_workflow
from ..imaging.imaging_arlexecute import subtract_list_arlexecute_workflow, \
    restore_list_arlexecute_workflow, deconvolve_list_arlexecute_workflow, residual_list_arlexecute_workflow
from ..skymodel.skymodel_arlexecute import predict_skymodel_list_arlexecute_workflow, \
    invert_skymodel_list_arlexecute_workflow


def mpccal_skymodel_list_arlexecute_workflow(vis_list, skymodel_list, context, vis_slices=1, facets=1,
                                             gcfcf=None, calibration_context='TG', do_selfcal=True, **kwargs):
    """Create graph for MPCCAL pipeline

    :param vis_list:
    :param skymodel_list:
    :param context: imaging context e.g. '2d'
    :param calibration_context: Sequence of calibration steps e.g. TGB
    :param do_selfcal: Do the selfcalibration?
    :param kwargs: Parameters for functions in components
    :return:
    """
    
    if gcfcf is None:
        gcfcf = [arlexecute.execute(create_pswf_convolutionfunction)(skymodel_list[0])]
    
    psf_imagelist = invert_skymodel_list_arlexecute_workflow(vis_list, skymodel_list, dopsf=True, context=context,
                                                             vis_slices=vis_slices, facets=facets, gcgcf=gcfcf,
                                                             **kwargs)
    
    model_vislist = [arlexecute.execute(copy_visibility, nout=1)(v, zero=True) for v in vis_list]
    
    cal_vis_list = [arlexecute.execute(copy_visibility, nout=1)(v) for v in vis_list]
    
    # Make the predicted visibilities, selfcalibrate against it correcting the gains, then
    # form the residual visibility, then make the residual image
    model_vislist = predict_skymodel_list_arlexecute_workflow(model_vislist, skymodel_list,
                                                              context=context, vis_slices=vis_slices, facets=facets,
                                                              gcgcf=gcfcf, docal=True, **kwargs)
    cal_vis_list, gt_list = calibrate_list_arlexecute_workflow(cal_vis_list, model_vislist,
                                                         calibration_context=calibration_context, **kwargs)
    residual_vislist = subtract_list_arlexecute_workflow(cal_vis_list, model_vislist)
    residual_imagelist = invert_skymodel_list_arlexecute_workflow(residual_vislist, skymodel_list,
                                                                  context=context, dopsf=False,
                                                                  vis_slices=vis_slices, facets=facets, gcgcf=gcfcf,
                                                                  iteration=0, docal=True, **kwargs)
    
    deconvolve_skymodel_list, _ = deconvolve_list_arlexecute_workflow(residual_imagelist, psf_imagelist,
                                                                      skymodel_list,
                                                                      prefix='cycle 0',
                                                                      **kwargs)
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):
            model_vislist = predict_skymodel_list_arlexecute_workflow(model_vislist, deconvolve_skymodel_list,
                                                                      context=context, vis_slices=vis_slices,
                                                                      facets=facets, docal=True,
                                                                      gcgcf=gcfcf, **kwargs)
            cal_vis_list = [arlexecute.execute(copy_visibility, nout=1)(v) for v in vis_list]
            cal_vis_list, gt_list = calibrate_list_arlexecute_workflow(cal_vis_list, model_vislist,
                                                                 calibration_context=calibration_context,
                                                                 iteration=cycle, **kwargs)
            residual_vislist = subtract_list_arlexecute_workflow(cal_vis_list, model_vislist)
            residual_imagelist = invert_skymodel_list_arlexecute_workflow(residual_vislist, skymodel_list,
                                                                          context=context,
                                                                          vis_slices=vis_slices, facets=facets,
                                                                          docal=True, gcgcf=gcfcf, **kwargs)
            prefix = "cycle %d" % (cycle + 1)
            deconvolve_skymodel_list, _ = deconvolve_list_arlexecute_workflow(residual_imagelist, psf_imagelist,
                                                                              deconvolve_skymodel_list,
                                                                              prefix=prefix,
                                                                              **kwargs)
    residual_imagelist = residual_list_arlexecute_workflow(cal_vis_list, deconvolve_skymodel_list, context=context,
                                                           vis_slices=vis_slices, facets=facets, gcgcf=gcfcf, **kwargs)
    restore_imagelist = restore_list_arlexecute_workflow(deconvolve_skymodel_list, psf_imagelist, residual_imagelist)
    return arlexecute.execute((deconvolve_skymodel_list, residual_imagelist, restore_imagelist, gt_list))
