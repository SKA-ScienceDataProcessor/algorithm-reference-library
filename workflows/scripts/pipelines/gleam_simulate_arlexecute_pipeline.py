# coding: utf-8

# # Pipeline processing using Dask
# 
# This notebook demonstrates the continuum imaging and ICAL pipelines.

# In[ ]:


import os
import sys

sys.path.append(os.path.join('..', '..'))

from data_models.parameters import arl_path

results_dir = arl_path('test_results')

from matplotlib import pylab

pylab.rcParams['figure.figsize'] = (12.0, 12.0)
pylab.rcParams['image.cmap'] = 'rainbow'

import numpy
import pprint
import logging

from astropy.coordinates import SkyCoord
from astropy import units as u

from data_models.memory_data_models import SkyModel
from data_models.polarisation import PolarisationFrame
from data_models.data_model_helpers import export_skymodel_to_hdf5, export_blockvisibility_to_hdf5

from wrappers.arlexecute.simulation.testing_support import create_low_test_image_from_gleam
from wrappers.arlexecute.imaging.base import advise_wide_field

from workflows.arlexecute.imaging.imaging_arlexecute import predict_list_arlexecute_workflow
from workflows.arlexecute.simulation.simulation_arlexecute import simulate_list_arlexecute_workflow, corrupt_list_arlexecute_workflow
from wrappers.arlexecute.execution_support.dask_init import get_dask_Client
from wrappers.arlexecute.execution_support.arlexecute import arlexecute

pp = pprint.PrettyPrinter()


def init_logging():
    logging.basicConfig(filename='%s/gleam-pipeline.log' % results_dir,
                        filemode='a',
                        format='%(thread)s %(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


if __name__ == '__main__':
    log = logging.getLogger()
    logging.info("Starting gleam_simulate_arlexecute_pipeline")
    
    arlexecute.set_client(get_dask_Client())
    arlexecute.run(init_logging)
    
    # We create a graph to make the visibility. The parameter rmax determines the distance of the furthest antenna/stations used. All over parameters are determined from this number.
    
    # In[ ]:
    
    nfreqwin = 7
    ntimes = 11
    rmax = 300.0
    frequency = numpy.linspace(0.9e8, 1.1e8, nfreqwin)
    channel_bandwidth = numpy.array(nfreqwin * [frequency[1] - frequency[0]])
    times = numpy.linspace(-numpy.pi / 3.0, numpy.pi / 3.0, ntimes)
    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
    
    vis_list = simulate_list_arlexecute_workflow('LOWBD2',
                                                 rmax=rmax,
                                                 frequency=frequency,
                                                 channel_bandwidth=channel_bandwidth,
                                                 times=times,
                                                 phasecentre=phasecentre,
                                                 order='frequency')
    print('%d elements in vis_list' % len(vis_list))
    log.info('About to make visibility')
    vis_list = arlexecute.compute(vis_list, sync=True)
    
    print(vis_list[0])

    
    # In[ ]:
    
    wprojection_planes = 1
    advice_low = advise_wide_field(vis_list[0], guard_band_image=8.0, delA=0.02, wprojection_planes=wprojection_planes)
    
    advice_high = advise_wide_field(vis_list[-1], guard_band_image=8.0, delA=0.02,
                                    wprojection_planes=wprojection_planes)
    
    vis_slices = advice_low['vis_slices']
    npixel = advice_high['npixels2']
    cellsize = min(advice_low['cellsize'], advice_high['cellsize'])
    
    # Now make a graph to fill with a model drawn from GLEAM
    
    # In[ ]:
    
    gleam_model = [arlexecute.execute(create_low_test_image_from_gleam)(npixel=npixel,
                                                                        frequency=[frequency[f]],
                                                                        channel_bandwidth=[channel_bandwidth[f]],
                                                                        cellsize=cellsize,
                                                                        phasecentre=phasecentre,
                                                                        polarisation_frame=PolarisationFrame("stokesI"),
                                                                        flux_limit=1.0,
                                                                        applybeam=True)
                   for f, freq in enumerate(frequency)]
    log.info('About to make GLEAM model')
    gleam_model = arlexecute.compute(gleam_model, sync=True)
    gleam_skymodel = SkyModel(images=gleam_model)
    export_skymodel_to_hdf5(gleam_skymodel, 'gleam_simulation_skymodel.hdf')
    future_gleam_model = arlexecute.scatter(gleam_model)
    
    # In[ ]:
    
    log.info('About to run predict to get predicted visibility')
    future_vis_graph = arlexecute.scatter(vis_list)
    predicted_vislist = predict_list_arlexecute_workflow(future_vis_graph, gleam_model, context='wstack', vis_slices=vis_slices)
    corrupted_vislist = corrupt_list_arlexecute_workflow(predicted_vislist, phase_error=1.0)
    log.info('About to run corrupt to get corrupted visibility')
    corrupted_vislist = arlexecute.compute(corrupted_vislist, sync=True)
    
    export_blockvisibility_to_hdf5(corrupted_vislist, 'gleam_simulation_vislist.hdf')
    
    arlexecute.close()