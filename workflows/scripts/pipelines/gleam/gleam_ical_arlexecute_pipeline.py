# coding: utf-8

# # Pipeline processing using Dask
# 
# This demonstrates the ICAL pipeline.

# In[ ]:


import os
import sys

from data_models.parameters import arl_path

results_dir = arl_path('test_results')

from matplotlib import pylab

pylab.rcParams['figure.figsize'] = (12.0, 12.0)
pylab.rcParams['image.cmap'] = 'rainbow'

from matplotlib import pyplot as plt

from data_models.polarisation import PolarisationFrame
from data_models.data_model_helpers import import_blockvisibility_from_hdf5

from processing_components.calibration.calibration_control import create_calibration_controls
from processing_components.image import show_image, export_image_to_fits, qa_image
from processing_components.imaging import create_image_from_visibility

from workflows.arlexecute.pipelines.pipeline_arlexecute import ical_list_arlexecute_workflow

from wrappers.arlexecute.execution_support.arlexecute import arlexecute

import pprint

import logging


def init_logging():
    logging.basicConfig(filename='%s/gleam-pipeline.log' % results_dir,
                        filemode='a',
                        format='%(thread)s %(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


if __name__ == '__main__':
    pp = pprint.PrettyPrinter()
    
    log = logging.getLogger()
    logging.info("Starting ical_list_arlexecute_workflow-pipeline")
    
    arlexecute.set_client(use_dask=True)
    arlexecute.run(init_logging)

    # Load data from previous simulation
    vislist = import_blockvisibility_from_hdf5('gleam_simulation_vislist.hdf')
    ntimes = len(vislist[0].time)

    cellsize = 0.001
    npixel = 1024
    pol_frame = PolarisationFrame("stokesI")
    
    model_list = [arlexecute.execute(create_image_from_visibility)(v, npixel=1024, cellsize=cellsize,
                                                                   polarisation_frame=pol_frame)
                  for v in vislist]
    model_list = arlexecute.persist(model_list)
    
    controls = create_calibration_controls()
    
    controls['T']['first_selfcal'] = 1
    controls['G']['first_selfcal'] = 3
    controls['B']['first_selfcal'] = 4
    
    controls['T']['timeslice'] = 'auto'
    controls['G']['timeslice'] = 'auto'
    controls['B']['timeslice'] = 1e5
    
    pp.pprint(controls)
    

    
    vislist = arlexecute.scatter(vislist)
    ical_list = ical_list_arlexecute_workflow(vislist,
                                              model_imagelist=model_list,
                                              context='wstack',
                                              calibration_context='TG',
                                              controls=controls,
                                              scales=[0, 3, 10], algorithm='mmclean',
                                              nmoment=3, niter=1000,
                                              fractional_threshold=0.1,
                                              threshold=0.1, nmajor=5, gain=0.25,
                                              deconvolve_facets=8,
                                              deconvolve_overlap=32,
                                              deconvolve_taper='tukey',
                                              vis_slices=ntimes,
                                              timeslice='auto',
                                              global_solution=False,
                                              psf_support=64,
                                              do_selfcal=True)
    

    
    log.info('About to run ical_list_serial_workflow')
    result = arlexecute.compute(ical_list, sync=True)
    deconvolved = result[0][0]
    residual = result[1][0]
    restored = result[2][0]
    arlexecute.close()

    show_image(deconvolved, title='Clean image', cm='Greys', vmax=0.1, vmin=-0.01)
    print(qa_image(deconvolved, context='Clean image'))
    plt.show()
    export_image_to_fits(deconvolved, '%s/gleam_ical_arlexecute_deconvolved.fits' % (results_dir))
    
    show_image(restored, title='Restored clean image', cm='Greys', vmax=0.1, vmin=-0.01)
    print(qa_image(restored, context='Restored clean image'))
    plt.show()
    export_image_to_fits(restored, '%s/gleam_ical_arlexecute_restored.fits' % (results_dir))
    
    show_image(residual[0], title='Residual clean image', cm='Greys', vmax=0.1, vmin=-0.01)
    print(qa_image(residual[0], context='Residual clean image'))
    plt.show()
    export_image_to_fits(residual[0], '%s/gleam_ical_arlexecute_residual.fits' % (results_dir))




