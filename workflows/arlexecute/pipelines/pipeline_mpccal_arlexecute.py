""" Pipeline functions. SDP standard pipelines expressed as functions.
"""

import logging

from processing_library.image.operations import create_empty_image_like
from workflows.arlexecute.skymodel.skymodel_arlexecute import predict_skymodel_list_arlexecute_workflow, \
    invert_skymodel_list_arlexecute_workflow, extract_datamodels_skymodel_list_arlexecute_workflow
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.image.operations import qa_image
from wrappers.arlexecute.skymodel.operations import update_skymodel_from_image, update_skymodel_from_gaintables, \
    calculate_skymodel_equivalent_image
from ..calibration.calibration_arlexecute import calibrate_list_arlexecute_workflow
from ..imaging.imaging_arlexecute import invert_list_arlexecute_workflow, \
    deconvolve_list_arlexecute_workflow
from ..skymodel.skymodel_arlexecute import convolve_skymodel_list_arlexecute_workflow

log = logging.getLogger(__name__)


def mpccal_skymodel_list_arlexecute_workflow(visobs, model, theta_list, nmajor=10, context='2d',
                                             mpccal_progress=None, **kwargs):
    """Run MPC
    
    :param visobs:
    :param theta_list:
    :param niter: Number of iterations
    :param context: Imaging context
    :return:
    """
    
    model = create_empty_image_like(theta_list[0].image)
    
    future_vobs = arlexecute.scatter(visobs)
    future_model = arlexecute.scatter(model)

    psf_obs = invert_list_arlexecute_workflow([future_vobs], [future_model], context=context, dopsf=True)
    theta_list = arlexecute.scatter(theta_list)
    
    for iteration in range(nmajor):
        vdatamodel_list = predict_skymodel_list_arlexecute_workflow(future_vobs, theta_list, context=context,
                                                                    docal=True, **kwargs)
        vdatamodel_list = extract_datamodels_skymodel_list_arlexecute_workflow(future_vobs, vdatamodel_list)
        dirty_all_conv = convolve_skymodel_list_arlexecute_workflow(future_vobs, theta_list, context=context,
                                                                    docal=True, **kwargs)
        dirty_all_cal = invert_skymodel_list_arlexecute_workflow(vdatamodel_list, theta_list, context=context,
                                                                 docal=True, **kwargs)
        
        def diff_dirty(dcal, dconv):
            dcal[0].data -= dconv[0].data
            return dcal
        
        dirty_all_cal = [arlexecute.execute(diff_dirty, nout=1)(dirty_all_cal[i], dirty_all_conv[i])
                         for i in range(len(dirty_all_cal))]
        
        def make_residual(dcal, tl):
            res = create_empty_image_like(model)
            for i, d in enumerate(dcal):
                res.data += d[0].data * tl[i].mask.data
            return res
        
        residual = arlexecute.execute(make_residual, nout=1)(dirty_all_cal, theta_list)
        
        deconvolved = deconvolve_list_arlexecute_workflow([[residual, 0.0]], [psf_obs[0]],
                                                          [model], **kwargs)
        
        theta_list = \
            arlexecute.execute(update_skymodel_from_image, nout=len(theta_list))(theta_list,
                                                                                 deconvolved[0][0])
        
        vpredicted_list = predict_skymodel_list_arlexecute_workflow(future_vobs, theta_list, context=context,
                                                                    docal=True, **kwargs)
        vcalibrated, gaintable_list = calibrate_list_arlexecute_workflow(vdatamodel_list, vpredicted_list,
                                                                         calibration_context='T',
                                                                         iteration=0, global_solution=False,
                                                                         **kwargs)
        theta_list = arlexecute.execute(update_skymodel_from_gaintables, nout=len(theta_list))(theta_list,
                                                                                               gaintable_list,
                                                                                               calibration_context='T')
        
        if mpccal_progress is None:
            
            def mpccal_progress(res, tl_list, gt_list, it):
                
                log.info('Iteration %d' % it)

                log.info('Length of theta = %d' % len(tl_list))
                
                log.info(qa_image(res, context='Residual image: iteration %d' % it))
                
                combined_model = calculate_skymodel_equivalent_image(tl_list)
                log.info(qa_image(combined_model, context='Combined model: iteration %d' % it))
                
                return arlexecute.execute(tl_list)
            
        theta_list = arlexecute.execute(mpccal_progress, nout=len(theta_list))(residual, theta_list, gaintable_list,
                                                                            iteration)

        result = arlexecute.execute((theta_list, residual))

    return  result
