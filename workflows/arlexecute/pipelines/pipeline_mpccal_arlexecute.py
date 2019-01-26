""" Pipeline functions. SDP standard pipelines expressed as functions.
"""

import logging

import numpy

from processing_library.image.operations import create_empty_image_like
from workflows.arlexecute.skymodel.skymodel_arlexecute import predict_skymodel_list_arlexecute_workflow, \
    invert_skymodel_list_arlexecute_workflow, extract_datamodels_skymodel_list_arlexecute_workflow
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.skymodel.operations import update_skymodel_from_image, update_skymodel_from_gaintables
from ..calibration.calibration_arlexecute import calibrate_list_arlexecute_workflow
from ..imaging.imaging_arlexecute import invert_list_arlexecute_workflow, \
    deconvolve_list_arlexecute_workflow
from ..skymodel.skymodel_arlexecute import convolve_skymodel_list_arlexecute_workflow

log = logging.getLogger(__name__)


def mpccal_skymodel_list_arlexecute_workflow(visobs, model, theta_list, nmajor=10, context='2d',
                                             **kwargs):
    """Run MPC pipeline
    
    :param visobs:
    :param model: Model image
    :param theta_list: SkyModel i.e. theta in memo 97.
    :param nmajor: Number of major cycles
    :param context: Imaging context
    :param mpccal_progress: Function to display progress
    :return: Delayed tuple (theta_list, residual)
    """
    psf_obs = invert_list_arlexecute_workflow([visobs], [model], context=context, dopsf=True)

    for iteration in range(nmajor):
        # The E step of decoupling the data models
        vdatamodel_list = predict_skymodel_list_arlexecute_workflow(visobs, theta_list, context=context,
                                                                    docal=True, **kwargs)
        vdatamodel_list = extract_datamodels_skymodel_list_arlexecute_workflow(visobs, vdatamodel_list)
        
        # The M step: 1 - Update the models by deconvolving the residual image
        dirty_all_conv = convolve_skymodel_list_arlexecute_workflow(visobs, theta_list, context=context,
                                                                    docal=True, **kwargs)
        dirty_all_cal = invert_skymodel_list_arlexecute_workflow(vdatamodel_list, theta_list, context=context,
                                                                 docal=True, **kwargs)
        
        def diff_dirty(dcal, dconv):
            assert numpy.max(numpy.abs(dcal[0].data)) > 0.0, "before: dcal subimage is zero"
            dcal[0].data -= dconv[0].data
            assert numpy.max(numpy.abs(dcal[0].data)) > 0.0, "after: dcal subimage is zero"
            return dcal
        
        dirty_all_cal = [arlexecute.execute(diff_dirty, nout=1)(dirty_all_cal[i], dirty_all_conv[i])
                         for i in range(len(dirty_all_cal))]
        
        def make_residual(dcal, tl):
            res = create_empty_image_like(dcal[0][0])
            for i, d in enumerate(dcal):
                assert numpy.max(numpy.abs(d[0].data)) > 0.0, "Residual subimage is zero"
                assert numpy.max(numpy.abs(tl[i].mask.data)) > 0.0, "Mask image is zero"
                res.data += d[0].data * tl[i].mask.data
            assert numpy.max(numpy.abs(res.data)) > 0.0, "Residual image is zero"
            return res
        
        residual = arlexecute.execute(make_residual, nout=1)(dirty_all_cal, theta_list)
        
        deconvolved = deconvolve_list_arlexecute_workflow([(residual, 0.0)], [psf_obs[0]],
                                                          [model], **kwargs)
        
        theta_list = \
            arlexecute.execute(update_skymodel_from_image, nout=len(theta_list))(theta_list,
                                                                                 deconvolved[0][0])
        # The M step: 2 - Update the gaintables
        vpredicted_list = predict_skymodel_list_arlexecute_workflow(visobs, theta_list, context=context,
                                                                    docal=True, **kwargs)
        _, gaintable_list = calibrate_list_arlexecute_workflow(vdatamodel_list, vpredicted_list,
                                                               calibration_context='T',
                                                               iteration=0, global_solution=False,
                                                               **kwargs)
        theta_list = arlexecute.execute(update_skymodel_from_gaintables, nout=len(theta_list))(theta_list,
                                                                                               gaintable_list,
                                                                                               calibration_context='T')
        
        result = arlexecute.execute((theta_list, residual))
    
    return result
