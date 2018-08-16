# coding: utf-8

# # Pipeline processing using Dask
# 
# This demonstrates the ICAL pipeline.

# In[ ]:


import os
import sys

sys.path.append(os.path.join('..', '..'))

from data_models.parameters import arl_path

results_dir = arl_path('test_results')

from matplotlib import pylab

pylab.rcParams['figure.figsize'] = (12.0, 12.0)
pylab.rcParams['image.cmap'] = 'rainbow'

from matplotlib import pyplot as plt

from data_models.polarisation import PolarisationFrame
from data_models.data_model_helpers import import_blockvisibility_from_hdf5

from wrappers.serial.calibration.calibration_control import create_calibration_controls
from wrappers.serial.image.operations import show_image, export_image_to_fits, qa_image
from wrappers.serial.imaging.base import create_image_from_visibility

from workflows.serial.pipelines.pipeline_serial import ical_list_serial_workflow

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
    logging.info("Starting ical_list_serial_workflow-pipeline")
    
    init_logging()

    # Load data from previous simulation
    vislist = import_blockvisibility_from_hdf5('gleam_simulation_vislist.hdf')
    
    print(vislist[0])
    
    cellsize = 0.001
    npixel = 1024
    pol_frame = PolarisationFrame("stokesI")
    
    model_list = [create_image_from_visibility(v, npixel=1024, cellsize=cellsize,
                                                                   polarisation_frame=pol_frame)
                  for v in vislist]
    
    controls = create_calibration_controls()
    
    controls['T']['first_selfcal'] = 1
    controls['G']['first_selfcal'] = 3
    controls['B']['first_selfcal'] = 4
    
    controls['T']['timescale'] = 'auto'
    controls['G']['timescale'] = 'auto'
    controls['B']['timescale'] = 1e5
    
    pp.pprint(controls)
    
    # In[ ]:
    
    ntimes = len(vislist[0].time)
    result = ical_list_serial_workflow(vislist,
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
    
    # In[ ]:
    
    log.info('About to run ical_list_serial_workflow')
    deconvolved = result[0][0]
    residual = result[1][0]
    restored = result[2][0]

    show_image(deconvolved, title='Clean image', cm='Greys', vmax=0.1, vmin=-0.01)
    print(qa_image(deconvolved, context='Clean image'))
    plt.show()
    export_image_to_fits(deconvolved, '%s/gleam_ical_serial_deconvolved.fits' % (results_dir))
    
    show_image(restored, title='Restored clean image', cm='Greys', vmax=0.1, vmin=-0.01)
    print(qa_image(restored, context='Restored clean image'))
    plt.show()
    export_image_to_fits(restored, '%s/gleam_ical_serial_restored.fits' % (results_dir))
    
    show_image(residual[0], title='Residual clean image', cm='Greys', vmax=0.1, vmin=-0.01)
    print(qa_image(residual[0], context='Residual clean image'))
    plt.show()
    export_image_to_fits(residual[0], '%s/gleam_ical_serial_residual.fits' % (results_dir))




