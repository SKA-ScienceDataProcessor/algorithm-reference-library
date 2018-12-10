# coding: utf-8


from data_models.parameters import arl_path

results_dir = arl_path('test_results')
dask_dir = arl_path('test_results/dask-work-space')

import numpy
import logging

from astropy.coordinates import SkyCoord
from astropy import units as u

from data_models.memory_data_models import SkyModel
from data_models.polarisation import PolarisationFrame
from data_models.data_model_helpers import export_skymodel_to_hdf5, export_blockvisibility_to_hdf5

from wrappers.arlexecute.simulation.testing_support import create_low_test_image_from_gleam
from wrappers.arlexecute.imaging.base import advise_wide_field

from workflows.arlexecute.imaging.imaging_arlexecute import predict_list_arlexecute_workflow
from workflows.arlexecute.simulation.simulation_arlexecute import simulate_list_arlexecute_workflow, \
    corrupt_list_arlexecute_workflow
from processing_library.image.operations import create_empty_image_like

from wrappers.arlexecute.execution_support.arlexecute import arlexecute


def init_logging():
    logging.basicConfig(filename='%s/ska-pipeline.log' % results_dir,
                        filemode='a',
                        format='%%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


if __name__ == '__main__':
    log = logging.getLogger()
    print("Starting ska-pipelines pipeline")
    
    arlexecute.set_client(use_dask=True, threads_per_worker=1, memory_limit=4e9,
                          local_dir=dask_dir)
    print(arlexecute.client)
    arlexecute.run(init_logging)
    
    # We create a graph to make the visibility. The parameter rmax determines the distance of the
    # furthest antenna/stations used. All over parameters are determined from this number.
    
    nfreqwin = 41
    ntimes = 5
    rmax = 750.0
    centre = nfreqwin // 2
    
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
    print('About to make visibility')
    vis_list = arlexecute.persist(vis_list)
    
    # The vis data are on the workers so we run the advice function on the workers
    # without transfering the data back to the host.
    advice_list = [arlexecute.execute(advise_wide_field)(v, guard_band_image=8.0, delA=0.02,
                                                         wprojection_planes=1)
                   for _, v in enumerate(vis_list)]
    advice_list = arlexecute.compute(advice_list, sync=True)
    
    advice_low = advice_list[0]
    advice_high = advice_list[-1]
    print(advice_list[centre])
    
    vis_slices = advice_low['vis_slices']
    npixel = advice_high['npixels2']
    cellsize = min(advice_low['cellsize'], advice_high['cellsize'])
    
    # Now make a graph to fill with a model drawn from GLEAM
    
    # In[ ]:
    
    dprepb_model = [arlexecute.execute(create_low_test_image_from_gleam)(npixel=npixel,
                                                                         frequency=[frequency[f]],
                                                                         channel_bandwidth=[channel_bandwidth[f]],
                                                                         cellsize=cellsize,
                                                                         phasecentre=phasecentre,
                                                                         polarisation_frame=PolarisationFrame(
                                                                             "stokesI"),
                                                                         flux_limit=3.0,
                                                                         applybeam=True)
                    for f, freq in enumerate(frequency)]
    # Put the model on the cluster
    dprepb_model = arlexecute.persist(dprepb_model)
    
    print('About to make GLEAM model')
    zero_model = [arlexecute.execute(create_empty_image_like)(im) for im in dprepb_model]
    zero_model = arlexecute.compute(zero_model, sync=True)
    zero_skymodel = SkyModel(images=zero_model)
    export_skymodel_to_hdf5(zero_skymodel, arl_path('%s/ska-pipeline_simulation_skymodel.hdf' % results_dir))
    
    # In[ ]:
    
    #    vis_list = arlexecute.scatter(vis_list)
    
    wstack = True
    if wstack:
        print('Using w stack with %d slices' % vis_slices)
        predicted_vislist = predict_list_arlexecute_workflow(vis_list, dprepb_model, context='wstack',
                                                             vis_slices=vis_slices)
    else:
        print('Using timeslicing with %d slices' % ntimes)
        predicted_vislist = predict_list_arlexecute_workflow(vis_list, dprepb_model, context='timeslice',
                                                             vis_slices=ntimes)
    
    corrupted_vislist = corrupt_list_arlexecute_workflow(predicted_vislist, phase_error=1.0, seed=180555)
    
    export_list = [arlexecute.execute(export_blockvisibility_to_hdf5)
                   (corrupted_vislist[v], arl_path('%s/ska-pipeline_simulation_vislist_%d.hdf' % (results_dir, v)))
                   for v, _ in enumerate(corrupted_vislist)]
    
    print('About to run predict and corrupt to get corrupted visibility, and write files')
    arlexecute.compute(export_list, sync=True)
    
    arlexecute.close()
