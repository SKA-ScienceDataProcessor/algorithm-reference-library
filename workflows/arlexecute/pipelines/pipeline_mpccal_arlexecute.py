""" Pipeline functions. SDP standard pipelines expressed as functions.
"""

from data_models.parameters import get_parameter
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.griddata.kernels import create_pswf_convolutionfunction
from wrappers.arlexecute.visibility.base import copy_visibility
from ..calibration.calibration_arlexecute import calibrate_list_arlexecute_workflow
from ..imaging.imaging_arlexecute import subtract_list_arlexecute_workflow, invert_list_arlexecute_workflow, \
    restore_list_arlexecute_workflow, deconvolve_list_arlexecute_workflow, residual_list_arlexecute_workflow
from ..skymodel.skymodel_arlexecute import predict_skymodel_list_arlexecute_workflow, \
    invert_skymodel_list_arlexecute_workflow, convolve_skymodel_list_arlexecute_workflow
from operations import update_skymodel_from_image, update_skymodel_from_gaintables

from workflows.arlexecute.skymodel.skymodel_arlexecute import predict_skymodel_list_arlexecute_workflow, \
    invert_skymodel_list_arlexecute_workflow, convert_blockvisibility_to_visibility, \
    extract_datamodels_skymodel_list_arlexecute_workflow
from workflows.shared.imaging.imaging_shared import threshold_list

def mpccal_skymodel_list_arlexecute_workflow(obsvis, skymodel_list, context, vis_slices=1, facets=1,
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
        gcfcf = [arlexecute.execute(create_pswf_convolutionfunction)(skymodel_list[0].image)]
    
    psf_imagelist = invert_skymodel_list_arlexecute_workflow(obsvis, skymodel_list, dopsf=True, context=context,
                                                             vis_slices=vis_slices, facets=facets, gcgcf=gcfcf,
                                                             **kwargs)
    # Make a single model vis, this gets expanded into a list of datamodels by the predict
    model_vis = arlexecute.execute(copy_visibility, nout=1)(obsvis)
    model_vislist = predict_skymodel_list_arlexecute_workflow(model_vis, skymodel_list,
                                                              context=context, vis_slices=vis_slices, facets=facets,
                                                              gcgcf=gcfcf, docal=True, **kwargs)
    cal_vis_list = [arlexecute.execute(copy_visibility, nout=1)(obsvis) for v in model_vislist]
    
    cal_vis_list, gt_list = calibrate_list_arlexecute_workflow(cal_vis_list, model_vislist,
                                                               calibration_context=calibration_context, **kwargs)
    residual_vislist = subtract_list_arlexecute_workflow(cal_vis_list, model_vislist)
    residual_imagelist = invert_skymodel_list_arlexecute_workflow(residual_vislist, skymodel_list,
                                                                  context=context, dopsf=False,
                                                                  vis_slices=vis_slices, facets=facets, gcgcf=gcfcf,
                                                                  iteration=0, docal=True, **kwargs)
    
    deconvolve_skymodel_list, _ = deconvolve_list_arlexecute_workflow(residual_imagelist, psf_imagelist,
                                                                      skymodel_list[0].image,
                                                                      prefix='cycle 0',
                                                                      **kwargs)
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):
            model_vislist = predict_skymodel_list_arlexecute_workflow(model_vislist, deconvolve_skymodel_list,
                                                                      context=context, vis_slices=vis_slices,
                                                                      facets=facets, docal=True,
                                                                      gcgcf=gcfcf, **kwargs)
            cal_vis_list = [arlexecute.execute(copy_visibility, nout=1)(v) for v in model_vislist]
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
    model_vislist = predict_skymodel_list_arlexecute_workflow(model_vislist, deconvolve_skymodel_list,
                                                              context=context, vis_slices=vis_slices,
                                                              facets=facets, docal=True,
                                                              gcgcf=gcfcf, **kwargs)
    residual_vislist = subtract_list_arlexecute_workflow(cal_vis_list, model_vislist)
    residual_imagelist = invert_skymodel_list_arlexecute_workflow(residual_vislist, skymodel_list,
                                                                  context=context,
                                                                  vis_slices=vis_slices, facets=facets,
                                                                  docal=True, gcgcf=gcfcf, **kwargs)
    restore_imagelist = restore_list_arlexecute_workflow(deconvolve_skymodel_list, psf_imagelist, residual_imagelist)
    return arlexecute.execute((deconvolve_skymodel_list, residual_imagelist, restore_imagelist, gt_list))

def mpccal_arlexecute_workflow(visobs, model, theta_list, niter=10, context='2d', threshold=0.0,
                               fractional_threshold=0.1, calibration_context='T', **kwargs):
    """Run MPC
    
    :param visobs:
    :param theta_list:
    :param niter: Number of iterations
    :param context: Imaging context
    :return:
    """

    future_Vobs = arlexecute.scatter(visobs)
    
    psf_obs = invert_list_arlexecute_workflow([visobs], model, dopsf=True, context=context, **kwargs)
    
    for iteration in range(niter):
    
        Vdatamodel_list = predict_skymodel_list_arlexecute_workflow(future_Vobs, theta_list, context='2d', docal=True)
        Vdatamodel_list = extract_datamodels_skymodel_list_arlexecute_workflow(future_Vobs, Vdatamodel_list)
        
        dirty_all_conv = convolve_skymodel_list_arlexecute_workflow(future_Vobs, theta_list, context='2d', docal=True)
        dirty_all_cal = invert_skymodel_list_arlexecute_workflow(Vdatamodel_list, theta_list, context='2d', docal=True)
        
        def sub_images(dcal, dconv):
            dcal[0].data -= dconv[0].data
            return dcal
          
        dirty_all_cal = arlexecute.execute(sub_images)(dirty_all_cal, dirty_all_conv)
        
        peak = arlexecute.execute(threshold_list)(dirty_all_cal, threshold, fractional_threshold)
        
        deconvolved_list = [deconvolve_list_arlexecute_workflow([dirty_all_cal[ism]], [psf_obs[0]],
                                                                      [model], mask=sm.mask, algorithm='msclean',
                                                                      scales=[0, 3, 10], niter=100,
                                                                      fractional_threshold=0.3, threshold=0.3 * peak,
                                                                      gain=0.1,
                                                                      psf_support=128, deconvolve_facets=8,
                                                                      deconvolve_overlap=16,
                                                                      deconvolve_taper='tukey')
                            for ism, sm in enumerate(theta_list)]
            
        def add_deconvolved(th, deconvolved):
            th.image.data += deconvolved[0].data
            return th

        def multiply_gaintable(th, gaintable):
            th.gaintable.gain *= gaintable[calibration_context].gain
            return th

        theta_list = [arlexecute.execute(add_deconvolved)(theta_list[ith], deconvolved_list[ith])
                      for ith, _ in enumerate(theta_list)]
        
        Vpredicted_list = predict_skymodel_list_arlexecute_workflow(future_Vobs, theta_list, context='2d', docal=True)
        result = calibrate_list_arlexecute_workflow(Vdatamodel_list, Vpredicted_list,
                                                    calibration_context='T',
                                                    iteration=0, global_solution=False)
        theta_list = [arlexecute.execute(multiply_gaintable)(th.gaintable, result[1][ith])
                      for ith, th in enumerate(theta_list)]
        
        return theta_list