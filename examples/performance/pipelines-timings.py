# # Pipeline processing using Dask
# 
# This notebook demonstrates the continuum imaging and ICAL pipelines.

import os
import sys
import time
import socket
sys.path.append(os.path.join('..', '..'))

import numpy

from arl.graphs.dask_init import get_dask_Client, get_nodes
from astropy.coordinates import SkyCoord
from astropy import units as u
from arl.data.polarisation import PolarisationFrame
from arl.image.operations import qa_image
from arl.imaging import create_image_from_visibility, advise_wide_field
from arl.graphs.graphs import create_invert_wstack_graph, \
    create_predict_graph, create_residual_graph, create_invert_graph, \
    create_predict_wstack_graph, create_residual_wstack_graph, create_invert_facet_wstack_graph, \
    create_predict_facet_wstack_graph, create_residual_facet_wstack_graph, \
    compute_list, create_invert_timeslice_graph, create_predict_timeslice_graph, \
    create_residual_timeslice_graph, create_deconvolve_facet_graph
from arl.util.graph_support import create_simulate_vis_graph, create_corrupt_vis_graph
from arl.util.testing_support import create_low_test_image_from_gleam, create_low_test_beam
from arl.pipelines.graphs import create_ical_pipeline_graph
from arl.graphs.vis import simple_vis

import logging

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

def git_hash():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", 'HEAD'])
    except:
        return "unknown"


def trial_case(seed=180555, context='', processor='wstack', n_workers=8, threads_per_worker=1,
               processes=True, order='frequency', nfreqwin=7, ntimes=3, rmax=750.0,
               facets=1, wprojection_planes=1, plot_graphs=False, do_ical=True, **kwargs):
    """ Single trial for performance-timings
    
    Simulates visibilities from GLEAM including phase errors
    Makes dirty image and PSF
    Runs ICAL pipeline
    
    The results are in a dictionary:
    
    'context': input - a string describing concisely the purpose of the test
    'time overall',  overall execution time (s)
    'time create gleam', time to create GLEAM prediction graph
    'time predict', time to execute GLEAM prediction graph
    'time corrupt', time to corrupt data
    'time invert', time to make dirty image
    'time psf invert', time to make PSF
    'time ICAL graph', time to create ICAL graph
    'time ICAL', time to execute ICAL graph
    'processor', type of imaging e.g. 'wstack'
    'n_workers', number of workers to create
    'threads_per_worker',
    'processes', 'order', Ordering of data
    'nfreqwin', Number of frequency windows in simulation
    'ntimes', Number of hour angles in simulation
    'rmax', Maximum radius of stations used in simulation (m)
    'facets', Number of facets in deconvolution and imaging
    'wprojection_planes', Number of wprojection planes
    'vis_slices', Number of visibility slices (per Visibbility)
    'npixel', Number of pixels in image
    'cellsize', Cellsize in radians
    'seed', Random number seed
    'dirty_max', Maximum in dirty image
    'dirty_min', Minimum in dirty image
    'psf_max',
    'psf_min',
    'restored_max',
    'restored_min',
    'deconvolved_max',
    'deconvolved_min',
    'residual_max',
    'residual_min',
    'git_info', GIT hash (not definitive since local mods are possible)
    
    :param seed: Random number seed (used in gain simulations)
    :param context: String to track reason for test e.g. 'scaling'
    :param processor: Type of processor: '2d'|'timeslice'|'wstack'
    :param n_workers: Number of dask workers to use
    :param threads_per_worker: Number of threads per worker
    :param processes: Use processes instead of threads 'processes'|'threads'
    :param order: See create_simulate_vis_graph
    :param nfreqwin: See create_simulate_vis_graph
    :param ntimes: See create_simulate_vis_graph
    :param rmax: See create_simulate_vis_graph
    :param facets: Number of facets to use
    :param wprojection_planes: Number of wprojection planes to use
    :param kwargs:
    :return: results dictionary
    """
    results = {}
    
    numpy.random.seed(seed)
    results['seed'] = seed
    
    start_all = time.time()
    
    results['context'] = context
    results['processor'] = processor
    results['hostname'] = socket.gethostname()
    results['git_hash'] = git_hash()
    results['epoch'] = time.strftime("%Y-%m-%d %H:%M:%S")
 
    zerow = False
    print("Processor is %s" % processor)
    if processor == 'timeslice':
        c_invert_graph = create_invert_timeslice_graph
        c_residual_graph = create_residual_timeslice_graph
        c_predict_graph = create_predict_timeslice_graph
    elif processor == 'wstack':
        c_invert_graph = create_invert_wstack_graph
        c_residual_graph = create_residual_wstack_graph
        c_predict_graph = create_predict_wstack_graph
    elif processor == 'facet_wstack':
        c_invert_graph = create_invert_facet_wstack_graph
        c_residual_graph = create_residual_facet_wstack_graph
        c_predict_graph = create_predict_facet_wstack_graph
    elif processor == '2d':
        c_invert_graph = create_invert_graph
        c_residual_graph = create_residual_graph
        c_predict_graph = create_predict_graph
        zerow = True
    else:
        c_invert_graph = create_invert_facet_wstack_graph
        c_residual_graph = create_residual_facet_wstack_graph
        c_predict_graph = create_predict_facet_wstack_graph

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
    
    print("At start, configuration is {0!r}".format(results))
    
    # Parameters determining scale
    frequency = numpy.linspace(0.8e8, 1.2e8, nfreqwin)
    if nfreqwin > 1:
        channel_bandwidth = numpy.array(nfreqwin * [frequency[1] - frequency[0]])
    else:
        channel_bandwidth = numpy.array([1e6])
    times = numpy.linspace(-numpy.pi / 3.0, numpy.pi / 3.0, ntimes)
    
    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
    
    vis_graph_list = create_simulate_vis_graph('LOWBD2',
                                               frequency=frequency,
                                               channel_bandwidth=channel_bandwidth,
                                               times=times,
                                               phasecentre=phasecentre,
                                               order=order,
                                               format='blockvis',
                                               rmax=rmax,
                                               seed=seed,
                                               zerow=zerow)
    print('%d elements in vis_graph_list' % len(vis_graph_list))
    
    nodes = get_nodes()
    print('Distributing vis_graphs to nodes %s: ' % nodes)
    
    client = get_dask_Client(n_workers=n_workers, threads_per_worker=threads_per_worker,
                             processes=processes)
    vis_graph_list = client.compute(vis_graph_list, sync=True, workers=nodes, **kwargs)
    print("After creating vis_graph_list", client)

    # Find the best imaging parameters.
    wprojection_planes = 1
    advice_low = advise_wide_field(vis_graph_list[0], guard_band_image=4.0, delA=0.02, facets=facets,
                                   wprojection_planes=wprojection_planes)
    
    advice_high = advise_wide_field(vis_graph_list[-1], guard_band_image=4.0, delA=0.02, facets=facets,
                                    wprojection_planes=wprojection_planes)
    
    kernel=advice_low['kernel']
    
    npixel = advice_high['npixels2']
    cellsize = advice_high['cellsize']
    
    if processor == 'timeslice':
        vis_slices = ntimes
    elif processor == '2d':
        vis_slices = 1
        kernel = '2d'
    else:
        vis_slices = advice_low['vis_slices']
    
    results['vis_slices'] = vis_slices
    results['cellsize'] = cellsize
    results['npixel'] = npixel
    
    # Create a realistic image using GLEAM and apply the primary beam. We do this only at the centre frequeny
    start = time.time()
    gleam_model = create_low_test_image_from_gleam(npixel=npixel, frequency=[frequency[len(frequency)//2]],
                                                   channel_bandwidth=[channel_bandwidth[len(frequency)//2]],
                                                   cellsize=cellsize, phasecentre=phasecentre)
    beam = create_low_test_beam(gleam_model)
    gleam_model.data *= beam.data
    
    end = time.time()
    results['time create gleam'] = end - start
    print("Creating GLEAM model took %.2f seconds" % (end - start))
    
#    vis_graph_list = c_predict_graph(vis_graph_list, gleam_model, vis_slices=vis_slices, facets=facets, kernel=kernel)
    vis_graph_list = c_predict_graph(vis_graph_list, gleam_model, vis_slices=5, facets=facets, kernel=kernel)
    print("After prediction", client)
    if plot_graphs:
        simple_vis(vis_graph_list[0], 'predict_%s' % processor, format='svg')
    start = time.time()
    vis_graph_list = client.compute(vis_graph_list, sync=True, workers=nodes, **kwargs)

    end = time.time()
    results['time predict'] = end - start
    print("GLEAM model Visibility prediction took %.2f seconds" % (end - start))
    assert numpy.max(numpy.abs(vis_graph_list[0].vis)), "Predicted visibility is zero"

    # Corrupt the visibility for the GLEAM model
    if do_ical:
        vis_graph_list = create_corrupt_vis_graph(vis_graph_list, phase_error=1.0)
        start = time.time()
        vis_graph_list = client.compute(vis_graph_list, sync=True, workers=nodes, **kwargs)
        end = time.time()
        results['time corrupt'] = end - start
        print("After corrupt", client)
        print("Visibility corruption took %.2f seconds" % (end - start))
    
    # Create a template model image
    model = create_image_from_visibility(vis_graph_list[len(vis_graph_list) // 2],
                                         npixel=npixel, cellsize=cellsize,
                                         frequency=[frequency[len(frequency) // 2]],
                                         channel_bandwidth=[channel_bandwidth[len(frequency) // 2]],
                                         polarisation_frame=PolarisationFrame("stokesI"))

    psf_graph = c_invert_graph(vis_graph_list, model, vis_slices=vis_slices, facets=facets, dopsf=True, kernel=kernel)
    start = time.time()
    psf, sumwt = client.compute(psf_graph, sync=True, workers=nodes, **kwargs)
    end = time.time()
    results['time psf invert'] = end - start
    print("PSF invert took %.2f seconds" % (end - start))
    print("After psf", client)

    results['psf_max'] = qa_image(psf).data['max']
    results['psf_min'] = qa_image(psf).data['min']
    
    dirty_graph = c_invert_graph(vis_graph_list, model, vis_slices=vis_slices, facets=facets, kernel=kernel)
    if plot_graphs:
        simple_vis(dirty_graph, 'invert_%s' % processor, format='svg')
    start = time.time()
    dirty, sumwt = client.compute(dirty_graph, sync=True, workers=nodes, **kwargs)
    end = time.time()
    print("After dirty image", client)
    results['time invert'] = end - start
    print("Dirty image invert took %.2f seconds" % (end - start))
    print("Maximum in dirty image is ", numpy.max(numpy.abs(dirty.data)), ", sumwt is ", sumwt)
    results['dirty_max'] = qa_image(dirty).data['max']
    results['dirty_min'] = qa_image(dirty).data['min']
    
    # Create the ICAL pipeline to run 5 major cycles, starting selfcal at cycle 1. A global solution across all
    # frequencies (i.e. Visibilities) is performed.
    if do_ical:
        start = time.time()
        ical_graph = create_ical_pipeline_graph(vis_graph_list,
                                                model_graph=model,
                                                c_deconvolve_graph=create_deconvolve_facet_graph,
                                                c_invert_graph=c_invert_graph,
                                                c_predict_graph=c_predict_graph,
                                                c_residual_graph=c_residual_graph,
                                                vis_slices=vis_slices, facets=facets,
                                                algorithm='hogbom', niter=1000,
                                                fractional_threshold=0.1,
                                                threshold=0.1, nmajor=5,
                                                gain=0.1, first_selfcal=1,
                                                global_solution=True,
                                                kernel=kernel)
        end = time.time()
        results['time ICAL graph'] = end - start
        print("ICAL graph creation took %.2f seconds" % (end - start))
        
        # Execute the graph
        start = time.time()
        deconvolved, residual, restored = client.compute(ical_graph, workers=nodes, sync=True, **kwargs)
        end = time.time()
        print("After ICAL", client)
        
        results['time ICAL'] = end - start
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
    else:
        results['time ICAL graph'] = 0.0
        results['time ICAL'] = 0.0
        results['deconvolved_max'] = 0.0
        results['deconvolved_min'] = 0.0
        results['residual_max'] = 0.0
        results['residual_min'] = 0.0
        results['restored_max'] = 0.0
        results['restored_min'] = 0.0

    #
    client.shutdown()
    
    end_all = time.time()
    results['time overall'] = end_all - start_all
    
    return results

def n_worker_trials():
    # Dask can give the number of workers available
    client = get_dask_Client()
    nproc = client.cluster.n_workers // 2
    client.shutdown()
    trials = []
    while nproc > 0:
        trials.append(nproc)
        nproc = nproc // 2
    return trials

def guess_rmax(rmax_standard=600.0, memory_standard=17179869184):
    import psutil
    return rmax_standard * numpy.sqrt(psutil.virtual_memory().total / memory_standard)


def guess_nfreqwin():
    import multiprocessing
    return max(1, multiprocessing.cpu_count() - 1)


def write_results(filename, results):
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writerow(results)
        print(results)
        csvfile.close()


def write_header(filename, fieldnames):
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        csvfile.close()


if __name__ == '__main__':
    import csv
    import seqfile
    
    import sys
    print(len(sys.argv))
    
    if len(sys.argv) == 1:
        print("python pipelines-timeings.py NUMBER_WORKERS NUMBER_FREQ NUMBER_THREADS")
        exit()
        
    if len(sys.argv) > 1:
        n_workers = [int(sys.argv[1])]
    else:
        n_workers = n_worker_trials()

    if len(sys.argv) > 2:
        nfreqwin = int(sys.argv[2])
    else:
        nfreqwin = n_workers

    if len(sys.argv) > 3:
        threads_per_worker = int(sys.argv[3])
    else:
        threads_per_worker = 1

    print("Trying %s workers" % n_workers)
    print("Using %s threads per worker" % threads_per_worker)
    print("Defining %d frequency windows" % nfreqwin)
    
    fieldnames = ['context', 'time overall', 'time create gleam', 'time predict', 'time corrupt',
                  'time invert', 'time psf invert', 'time ICAL graph', 'time ICAL',
                  'processor', 'n_workers', 'threads_per_worker', 'processes', 'order',
                  'nfreqwin', 'ntimes', 'rmax', 'facets', 'wprojection_planes', 'vis_slices', 'npixel',
                  'cellsize', 'seed', 'dirty_max', 'dirty_min', 'psf_max', 'psf_min', 'restored_max',
                  'restored_min', 'deconvolved_max', 'deconvolved_min', 'residual_max', 'residual_min',
                  'hostname', 'git_hash', 'epoch']

    
    filename = seqfile.findNextFile(prefix='pipelines-timings_', suffix='.csv')
    print('Saving results to %s' % filename)
    
    rmax = 600.0
    ntimes = 70
    n_repeats = 10
    
    # Scaling?
    contexts = ['scaling', 'processor', 'processes/threads', 'repeat']
    contexts = ['scaling']

    print('Tests being run: %s' % contexts)

    write_header(filename, fieldnames)
    
    # Match the number of frequency windows to the number of workers: run time should be constant.
    if 'scaling' in contexts:
        for n_worker in n_workers:
            results = trial_case(context='scaling', n_workers=n_worker, rmax=rmax, processor='2d',
                                 threads_per_worker=threads_per_worker, nfreqwin=nfreqwin, ntimes=ntimes)
            write_results(filename, results)

    # Processor
    if 'processor' in contexts:
        for processor in ['2d', 'wstack', 'timeslice']:
            results = trial_case(context='processor', processor=processor, n_workers=max(n_workers),
                                rmax=rmax, threads_per_worker=threads_per_worker, nfreqwin=nfreqwin, ntimes=ntimes)
            write_results(filename, results)

    
    # Which is faster? Processes or threads. Tests indicate processes so we don't test
    if 'processes/threads' in contexts:
        for n_workers in [max(n_workers)]:
            for processes in [True, False]:
                results = trial_case(context='processes/threads', n_workers=n_workers, rmax=rmax,
                                     threads_per_worker=threads_per_worker, processes=processes, nfreqwin=nfreqwin, ntimes=ntimes)
                write_results(filename, results)
    
    # # Repeatability? Don't use all the workers possible.
    if 'repeat' in contexts:
        for trial in range(n_repeats):
            results = trial_case(context='repeat', threads_per_worker=threads_per_worker,
                                 n_workers=max(n_workers)//2, rmax=rmax,
                                nfreqwin=nfreqwin, ntimes=ntimes)
            write_results(filename, results)

    print('Exiting pipelines-timings')
    exit()
