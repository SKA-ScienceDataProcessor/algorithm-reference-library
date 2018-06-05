# coding: utf-8

# # Pipeline processing using Dask
# 
# This demonstrates the ICAL pipeline.

# In[ ]:


import os
import sys

import json
from jsonschema import validate

sys.path.append(os.path.join('..', '..'))

from data_models.parameters import arl_path

results_dir = arl_path('test_results')

from matplotlib import pylab

pylab.rcParams['figure.figsize'] = (12.0, 12.0)
pylab.rcParams['image.cmap'] = 'rainbow'

from matplotlib import pyplot as plt

from data_models.polarisation import PolarisationFrame
from data_models.data_model_helpers import import_blockvisibility_from_hdf5

from processing_components.image.operations import show_image, export_image_to_fits, qa_image
from processing_components.imaging.base import create_image_from_visibility

from processing_components.pipelines.pipeline_components import continuum_imaging_component

from processing_components.component_support.arlexecute import arlexecute

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
    
    with open(arl_path('json/arl_schema.json'), 'r') as file:
        schema = file.read()
    arl_schema = json.loads(schema)
    
    with open('gleam_sip.json', 'r') as file:
        config = json.loads(file.read())
    
    validate(config, arl_schema)
    
    pp.pprint(config)
    
    arlexecute.set_client(use_dask=config["execute"]["use_dask"])
    arlexecute.run(init_logging)
    
    # Load data from previous simulation
    vislist = import_blockvisibility_from_hdf5(config["inputs"]["vislist"])
    
    print(vislist[0])
    
    cellsize = 0.001
    npixel = config["image"]["npixel"]
    pol_frame = PolarisationFrame(config["image"]["polarisation_frame"])
    
    model_list = [arlexecute.execute(create_image_from_visibility)(v, npixel=npixel, cellsize=cellsize,
                                                                   polarisation_frame=pol_frame)
                  for v in vislist]
    
    future_vislist = arlexecute.scatter(vislist)
    ntimes = len(vislist[0].time)
    continuum_imaging_list = \
        continuum_imaging_component(future_vislist,
                                    model_imagelist=model_list,
                                    context=config["image"]["context"],
                                    scales=config["deconvolution"]["scales"],
                                    algorithm=config["deconvolution"]["algorithm"],
                                    nmoment=config["deconvolution"]["nmoments"],
                                    niter=config["deconvolution"]["niter"],
                                    fractional_threshold=config["deconvolution"]["fractional_threshold"],
                                    threshold=config["deconvolution"]["threshold"],
                                    nmajor=config["deconvolution"]["nmajor"],
                                    gain=config["deconvolution"]["gain"],
                                    deconvolve_facets=config["deconvolution"]["deconvolve_facets"],
                                    deconvolve_overlap=config["deconvolution"]["deconvolve_overlap"],
                                    deconvolve_taper=config["deconvolution"]["deconvolve_taper"],
                                    vis_slices=config["imaging"]["vis_slices"],
                                    psf_support=config["deconvolution"]["psf_support"])
    
    # In[ ]:
    
    log.info('About to run continuum imaging')
    result = arlexecute.compute(continuum_imaging_list, sync=True)
    deconvolved = result[0][0]
    residual = result[1][0]
    restored = result[2][0]
    arlexecute.close()
    
    show_image(deconvolved, title='Clean image', cm='Greys', vmax=0.1, vmin=-0.01)
    print(qa_image(deconvolved, context='Clean image'))
    plt.show()
    export_image_to_fits(deconvolved, '%s/%s' % (config["outputs"]["deconvolved"], results_dir))
    
    show_image(restored, title='Restored clean image', cm='Greys', vmax=0.1, vmin=-0.01)
    print(qa_image(restored, context='Restored clean image'))
    plt.show()
    export_image_to_fits(restored, '%s/%s' % (config["outputs"]["restored"], results_dir))
    
    show_image(residual[0], title='Residual clean image', cm='Greys', vmax=0.1, vmin=-0.01)
    print(qa_image(residual[0], context='Residual clean image'))
    plt.show()
    export_image_to_fits(residual, '%s/%s' % (config["outputs"]["residual"], results_dir))
