# # Pipeline processing using Dask
# 
# This notebook demonstrates the continuum imaging and ICAL pipelines.

import os
import sys
import time

results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

sys.path.append(os.path.join('..', '..'))

import numpy

from distributed import Client, LocalCluster

from astropy.coordinates import SkyCoord
from astropy import units as u
from arl.data.polarisation import PolarisationFrame
from arl.image.operations import qa_image
from arl.imaging import create_image_from_visibility, advise_wide_field
from arl.graphs.graphs import create_invert_facet_wstack_graph, create_deconvolve_facet_graph, \
    create_predict_facet_wstack_graph, \
    compute_list, create_invert_timeslice_graph, create_predict_timeslice_graph, \
    create_residual_timeslice_graph, create_residual_facet_wstack_graph
from arl.util.graph_support import create_simulate_vis_graph, create_low_test_image_from_gleam, \
    create_low_test_beam, create_corrupt_vis_graph
from arl.pipelines.graphs import create_ical_pipeline_graph

import logging

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def trial_case(seed=180555, context='', processor='wstack', n_workers=8, threads_per_worker=1,
               processes=True, order='frequency', nfreqwin=7, ntimes=3, rmax=750.0,
               facets=1, wprojection_planes=1, **kwargs):

    results = {}
    
    numpy.random.seed(seed)
    results['seed'] = seed
    
    start_all = time.time()

    results['context'] = context
    results['processor'] = processor
    if processor == 'timeslice':
        c_invert = create_invert_timeslice_graph
        c_residual = create_residual_timeslice_graph
        c_predict = create_predict_timeslice_graph
    else:
        c_invert = create_invert_facet_wstack_graph
        c_residual = create_residual_facet_wstack_graph
        c_predict = create_predict_facet_wstack_graph


    results['processor'] = processor
    results['n_workers'] = n_workers
    results['threads_per_worker'] = threads_per_worker
    results['processes'] = processes
    results['order'] = order
    results['nfreqwin'] = nfreqwin
    results['ntimes'] = ntimes
    results['rmax'] = rmax
    results['facets'] = facets
    results['wprojection_planes'] = wprojection_planes
    
    # Parameters determining scale
    frequency = numpy.linspace(0.8e8, 1.2e8, nfreqwin)
    channel_bandwidth = numpy.array(nfreqwin * [frequency[1] - frequency[0]])
    times = numpy.linspace(-numpy.pi / 3.0, numpy.pi / 3.0, ntimes)
    
    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
    
    vis_graph_list = create_simulate_vis_graph('LOWBD2',
                                               frequency=frequency,
                                               channel_bandwidth=channel_bandwidth,
                                               times=times,
                                               phasecentre=phasecentre,
                                               order=order,
                                               format='blockvis',
                                               rmax=rmax)
    print('%d elements in vis_graph_list' % len(vis_graph_list))
    
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker,
                           processes=processes)
    client = Client(cluster)
    print(client)
    
    vis_graph_list = compute_list(client, vis_graph_list)
    
    # Find the best imaging parameters.
    wprojection_planes = 1
    advice_low = advise_wide_field(vis_graph_list[0], guard_band_image=4.0, delA=0.02, facets=facets,
                                   wprojection_planes=wprojection_planes)
    
    advice_high = advise_wide_field(vis_graph_list[-1], guard_band_image=4.0, delA=0.02, facets=facets,
                                    wprojection_planes=wprojection_planes)
    
    npixel = advice_high['npixels2']
    cellsize = advice_high['cellsize']
    
    if processor == 'timeslice':
        vis_slices = ntimes
    else:
        vis_slices = advice_low['vis_slices']
        
    results['vis_slices'] = vis_slices
    results['cellsize'] = cellsize
    results['npixel'] = npixel

    # Create a realistic image using GLEAM and apply the primary beam
    start = time.time()
    gleam_model = create_low_test_image_from_gleam(npixel=npixel, frequency=frequency,
                                                   channel_bandwidth=channel_bandwidth,
                                                   cellsize=cellsize, phasecentre=phasecentre)
    beam = create_low_test_beam(gleam_model)
    gleam_model.data *= beam.data
    end = time.time()
    results['create_gleam'] = end - start
    print("Creating GLEAM model took %.2f seconds" % (end - start))
    
    # Predict the visibility for the GLEAM model
    vis_graph_list = c_predict(vis_graph_list, gleam_model, vis_slices=vis_slices, facets=facets)
    vis_graph_list = create_corrupt_vis_graph(vis_graph_list, phase_error=1.0)
    start = time.time()
    vis_graph_list = compute_list(client, vis_graph_list)
    end = time.time()
    results['predict'] = end - start
    print("Visibility prediction took %.2f seconds" % (end - start))
    
    # Create a template model image
    model = create_image_from_visibility(vis_graph_list[len(vis_graph_list) // 2],
                                         npixel=npixel, cellsize=cellsize,
                                         frequency=[frequency[len(frequency) // 2]],
                                         channel_bandwidth=[channel_bandwidth[len(frequency) // 2]],
                                         polarisation_frame=PolarisationFrame("stokesI"))
    
    dirty_graph = c_invert(vis_graph_list, model, vis_slices=vis_slices, facets=facets)
    start = time.time()
    future = client.compute(dirty_graph)
    dirty, sumwt = future.result()
    end = time.time()
    results['invert'] = end - start
    print("Invert took %.2f seconds" % (end - start))
    results['dirty_max'] = qa_image(dirty).data['max']
    results['dirty_min'] = qa_image(dirty).data['min']

    psf_graph = c_invert(vis_graph_list, model, vis_slices=vis_slices, facets=facets, dopsf=True)
    start = time.time()
    future = client.compute(psf_graph)
    psf, sumwt = future.result()
    end = time.time()
    results['psf_invert'] = end - start
    print("Invert took %.2f seconds" % (end - start))
    results['psf_max'] = qa_image(psf).data['max']
    results['psf_min'] = qa_image(psf).data['min']

    # Create the ICAL pipeline
    start = time.time()
    ical_graph = create_ical_pipeline_graph(vis_graph_list,
                                            model_graph=model,
                                            c_deconvolve_graph=create_deconvolve_facet_graph,
                                            c_invert_graph=c_invert,
                                            c_predict_vis_graph=c_predict,
                                            c_residual_graph=c_residual,
                                            vis_slices=vis_slices, facets=facets,
                                            algorithm='hogbom', niter=1000,
                                            fractional_threshold=0.1,
                                            threshold=0.1, nmajor=5,
                                            gain=0.1, first_selfcal=1,
                                            global_solution=True)
    end = time.time()
    results['ical_graph_creation'] = end - start
    print("ICAL graph creation took %.2f seconds" % (end - start))
    
    # Execute the graph
    start = time.time()
    future = client.compute(ical_graph)
    deconvolved, residual, restored = future.result()
    end = time.time()
    results['ical'] = end - start
    print("ICAL compute took %.2f seconds" % (end - start))
    qa = qa_image(deconvolved)
    results['deconvolved_max'] = qa.data['max']
    results['deconvolved_min'] = qa.data['min']
    qa = qa_image(residual[0])
    results['residual_max'] = qa.data['max']
    results['residual_min'] = qa.data['min']
    qa = qa_image(restored)
    results['restored_max'] = qa.data['max']
    results['restored_min'] = qa.data['min']
    #
    client.shutdown()
    cluster.close()
    
    end_all = time.time()
    results['overall'] = end_all - start_all
    
    return results


def processor_trials():
    
    import multiprocessing
    nproc = multiprocessing.cpu_count()
    trials = []
    while nproc > 0:
        trials.append(nproc)
        nproc = nproc // 2
    return trials

def guess_rmax(rmax_standard=1000.0, memory_standard=17179869184, use_fraction=0.4):
    import psutil
    return rmax_standard * numpy.sqrt(use_fraction*psutil.virtual_memory().total/memory_standard)

def guess_nfreqwin():
    import multiprocessing
    return max(1, multiprocessing.cpu_count() - 1)

if __name__ == '__main__':
    import csv
    import seqfile
    import socket

    fieldnames = ['seed', 'context', 'processor', 'n_workers', 'threads_per_worker', 'processes', 'order',
                  'nfreqwin', 'ntimes', 'rmax', 'facets', 'wprojection_planes', 'vis_slices', 'npixel',
                  'cellsize', 'overall', 'create_gleam', 'predict', 'invert', 'psf_invert', 'ical_graph_creation',
                  'ical', 'dirty_max', 'dirty_min', 'psf_max', 'psf_min', 'restored_max', 'restored_min',
                  'deconvolved_max', 'deconvolved_min', 'residual_max', 'residual_min']

    filename = seqfile.findNextFile(prefix='pipelines-timings_%s_' % socket.gethostname(), suffix='.csv')
    print('Saving results to %s' % filename)
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        nprocessors = processor_trials()
        
        # Processor
        for processor in ['timeslice', 'wstack']:
            results = trial_case(context='processor', processor=processor, n_workers=max(nprocessors),
                                 rmax=guess_rmax(), nfreqwin=guess_nfreqwin())
            writer.writerow(results)
            print(results)
            
        # Which is faster? Processes or threads
        for n_workers in [max(nprocessors)]:
            results = trial_case(context='processes/threads', n_workers=n_workers, rmax=guess_rmax(),
                                 nfreqwin=guess_nfreqwin())
            writer.writerow(results)
            results = trial_case(context='processes/threads', n_workers=n_workers, processes=False,
                                 rmax=guess_rmax(), nfreqwin=guess_nfreqwin())
            writer.writerow(results)

        # Repeatability?
        for trial in range(3):
            results = trial_case(context='repeatability', n_workers=max(nprocessors), rmax=guess_rmax(),
                                 nfreqwin=guess_nfreqwin())
            writer.writerow(results)

        # Scaling?
        for n_workers in processor_trials():
            results = trial_case(context='scaling', n_workers=n_workers, rmax=guess_rmax(),
                                 nfreqwin=guess_nfreqwin())
            writer.writerow(results)

        exit()
