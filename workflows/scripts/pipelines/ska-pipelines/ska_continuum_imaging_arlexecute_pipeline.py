# coding: utf-8

# # Pipeline processing using Dask

import os
import sys

from data_models.parameters import arl_path

results_dir = arl_path('test_results')
dask_dir = arl_path('test_results/dask-work-space')

from data_models.polarisation import PolarisationFrame
from data_models.data_model_helpers import import_blockvisibility_from_hdf5

from processing_components.image.operations import show_image, export_image_to_fits, qa_image
from processing_components.imaging.base import create_image_from_visibility

from workflows.arlexecute.pipelines.pipeline_arlexecute import continuum_imaging_list_arlexecute_workflow

from wrappers.arlexecute.execution_support.arlexecute import arlexecute

import logging

def init_logging():
    logging.basicConfig(filename='%s/ska-pipeline.log' % results_dir,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


if __name__ == '__main__':
    
    log = logging.getLogger()
    logging.info("Starting continuum imaging pipeline")
    
    arlexecute.set_client(use_dask=True, threads_per_worker=1, memory_limit=4e9,
                          local_dir=dask_dir)
    print(arlexecute.client)
    arlexecute.run(init_logging)
    
    nfreqwin = 41
    ntimes = 5
    rmax = 750.0
    centre = nfreqwin // 2
    
    # Load data from previous simulation
    vis_list = [arlexecute.execute(import_blockvisibility_from_hdf5)
                (arl_path('%s/ska-pipeline_simulation_vislist_%d.hdf' % (results_dir, v)))
                for v in range(nfreqwin)]
    print('Reading visibilities')
    vis_list = arlexecute.persist(vis_list)
    
    cellsize = 0.001
    npixel = 1024
    pol_frame = PolarisationFrame("stokesI")
    
    model_list = [arlexecute.execute(create_image_from_visibility)(v, npixel=npixel, cellsize=cellsize,
                                                                   polarisation_frame=pol_frame)
                  for v in vis_list]
    
    model_list = arlexecute.persist(model_list)
    
    continuum_imaging_list = continuum_imaging_list_arlexecute_workflow(vis_list, model_imagelist=model_list,
                                                                        context='wstack', vis_slices=51,
                                                                        scales=[0, 3, 10], algorithm='mmclean',
                                                                        nmoment=3, niter=1000,
                                                                        fractional_threshold=0.1,
                                                                        threshold=0.1, nmajor=5, gain=0.25,
                                                                        deconvolve_facets=8,
                                                                        deconvolve_overlap=32,
                                                                        deconvolve_taper='tukey',
                                                                        timeslice='auto',
                                                                        psf_support=64)
    
    log.info('About to run continuum imaging workflow')
    result = arlexecute.compute(continuum_imaging_list, sync=True)
    arlexecute.close()
    
    deconvolved = result[0][centre]
    residual = result[1][centre]
    restored = result[2][centre]
    
    print(qa_image(deconvolved, context='Clean image'))
    export_image_to_fits(deconvolved, '%s/ska-continuum-imaging_arlexecute_deconvolved.fits' % (results_dir))
    
    print(qa_image(restored, context='Restored clean image'))
    export_image_to_fits(restored, '%s/ska-continuum-imaging_arlexecute_restored.fits' % (results_dir))
    
    print(qa_image(residual[0], context='Residual clean image'))
    export_image_to_fits(residual[0], '%s/ska-continuum-imaging_arlexecute_residual.fits' % (results_dir))
