# # Pipeline processing using Dask
# 
# This notebook demonstrates the continuum imaging and ICAL pipelines.

import logging
import socket
import sys
import time

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from dask_init import get_dask_Client, findNodes
from data_models.polarisation import PolarisationFrame

from processing_components.component_support.arlexecute import arlexecute
from processing_components.components.imaging_components import predict_component, invert_component
from processing_components.components.pipeline_components import ical_component
from processing_components.components.support_components import simulate_component, corrupt_component
from processing_components.image.operations import export_image_to_fits, qa_image, smooth_image
from processing_components.imaging.base import create_image_from_visibility, advise_wide_field
from processing_components.util.testing_support import create_low_test_image_from_gleam
from processing_components.visibility.coalesce import convert_blockvisibility_to_visibility

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

def git_hash():
    """ Get the hash for this git repository.
    
    :return: string or "unknown"
    """
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", 'HEAD'])
    except Exception as excp:
        print(excp)
        return "unknown"


def trial_case(results, seed=180555, context='wstack', nworkers=8, threads_per_worker=1,
               processes=True, order='frequency', nfreqwin=7, ntimes=3, rmax=750.0,
               facets=1, wprojection_planes=1, use_dask=True):
    """ Single trial for performance-timings
    
    Simulates visibilities from GLEAM including phase errors
    Makes dirty image and PSF
    Runs ICAL pipeline
    
    The results are in a dictionary:
    
    'context': input - a string describing concisely the purpose of the test
    'time overall',  overall execution time (s)
    'time create gleam', time to create GLEAM prediction graph
    'time predict', time to execute GLEAM prediction graph
    'time corrupt', time to corrupt data_models
    'time invert', time to make dirty image
    'time psf invert', time to make PSF
    'time ICAL graph', time to create ICAL graph
    'time ICAL', time to execute ICAL graph
    'context', type of imaging e.g. 'wstack'
    'nworkers', number of workers to create
    'threads_per_worker',
    'nnodes', Number of nodes,
    'processes', 'order', Ordering of data_models
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
    
    :param results: Initial state
    :param seed: Random number seed (used in gain simulations)
    :param context: imaging context
    :param context: Type of context: '2d'|'timeslice'|'wstack'
    :param nworkers: Number of dask workers to use
    :param threads_per_worker: Number of threads per worker
    :param processes: Use processes instead of threads 'processes'|'threads'
    :param order: See simulate_component
    :param nfreqwin: See simulate_component
    :param ntimes: See simulate_component
    :param rmax: See simulate_component
    :param facets: Number of facets to use
    :param wprojection_planes: Number of wprojection planes to use
    :param use_dask: Use dask or immediate evaluation
    :param kwargs:
    :return: results dictionary
    """
    
    numpy.random.seed(seed)
    results['seed'] = seed
    
    start_all = time.time()
    
    results['context'] = context
    results['hostname'] = socket.gethostname()
    results['git_hash'] = git_hash()
    results['epoch'] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    zerow = False
    print("Context is %s" % context)
    
    results['nworkers'] = nworkers
    results['threads_per_worker'] = threads_per_worker
    results['processes'] = processes
    results['order'] = order
    results['nfreqwin'] = nfreqwin
    results['ntimes'] = ntimes
    results['rmax'] = rmax
    results['facets'] = facets
    results['wprojection_planes'] = wprojection_planes
    
    results['use_dask'] = use_dask
    
    print("At start, configuration is {0!r}".format(results))
    
    # Parameters determining scale
    frequency = numpy.linspace(0.8e8, 1.2e8, nfreqwin)
    if nfreqwin > 1:
        channel_bandwidth = numpy.array(nfreqwin * [frequency[1] - frequency[0]])
    else:
        channel_bandwidth = numpy.array([1e6])
    times = numpy.linspace(-numpy.pi / 3.0, numpy.pi / 3.0, ntimes)
    
    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
    
    if use_dask:
        arlexecute.set_client(get_dask_Client(n_workers=nworkers, threads_per_worker=threads_per_worker,
                                              processes=processes), use_dask=True)
        results['nnodes'] = len(numpy.unique(findNodes(arlexecute.client)))
        print("Defined %d workers on %d nodes" % (nworkers, results['nnodes']))
    else:
        arlexecute.set_client(use_dask=use_dask)
        results['nnodes'] = 1

    
    vis_list = simulate_component('LOWBD2',
                                  frequency=frequency,
                                  channel_bandwidth=channel_bandwidth,
                                  times=times,
                                  phasecentre=phasecentre,
                                  order=order,
                                  format='blockvis',
                                  rmax=rmax)
    
    print("****** Visibility creation ******")
    vis_list = arlexecute.compute(vis_list, sync=True)
    
    # Find the best imaging parameters.
    wprojection_planes = 1
    advice = advise_wide_field(convert_blockvisibility_to_visibility(vis_list[0]), guard_band_image=6.0,
                               delA=0.02,
                               facets=facets,
                               wprojection_planes=wprojection_planes,
                               oversampling_synthesised_beam=4.0)
    
    kernel = advice['kernel']
    
    npixel = advice['npixels2']
    cellsize = advice['cellsize']
    
    if context == 'timeslice':
        vis_slices = ntimes
    elif context == '2d':
        vis_slices = 1
        kernel = '2d'
    else:
        vis_slices = advice['vis_slices']
    
    results['vis_slices'] = vis_slices
    results['cellsize'] = cellsize
    results['npixel'] = npixel
    
    gleam_model_list = [arlexecute.execute(create_low_test_image_from_gleam)(npixel=npixel,
                                                                  frequency=[frequency[f]],
                                                                  channel_bandwidth=[channel_bandwidth[f]],
                                                                  cellsize=cellsize,
                                                                  phasecentre=phasecentre,
                                                                  polarisation_frame=PolarisationFrame("stokesI"),
                                                                  flux_limit=0.1,
                                                                  applybeam=True)
                        for f, freq in enumerate(frequency)]
    
    start = time.time()
    print("****** Starting GLEAM model creation ******")
    gleam_model_list = arlexecute.compute(gleam_model_list, sync=True)
    cmodel = smooth_image(gleam_model_list[0])
    export_image_to_fits(cmodel, "pipelines-timings-delayed-gleam_cmodel.fits")
    end = time.time()
    results['time create gleam'] = end - start
    print("Creating GLEAM model took %.2f seconds" % (end - start))
    
    vis_list = predict_component(vis_list, gleam_model_list, vis_slices=51, context=context,
                                 kernel=kernel)
    start = time.time()
    print("****** Starting GLEAM model visibility prediction ******")
    vis_list = arlexecute.compute(vis_list, sync=True)
    
    end = time.time()
    results['time predict'] = end - start
    print("GLEAM model Visibility prediction took %.2f seconds" % (end - start))
    
    # Corrupt the visibility for the GLEAM model
    print("****** Visibility corruption ******")
    vis_list = corrupt_component(vis_list, phase_error=1.0)
    start = time.time()
    vis_list = arlexecute.compute(vis_list, sync=True)
    
    end = time.time()
    results['time corrupt'] = end - start
    print("Visibility corruption took %.2f seconds" % (end - start))
    
    # Create an empty model image
    model_list = [arlexecute.execute(create_image_from_visibility)(vis_list[f],
                                                        npixel=npixel, cellsize=cellsize,
                                                        frequency=[frequency[f]],
                                                        channel_bandwidth=[channel_bandwidth[f]],
                                                        polarisation_frame=PolarisationFrame("stokesI"))
                  for f, freq in enumerate(frequency)]
    model_list = arlexecute.compute(model_list, sync=True)
    
    psf_list = invert_component(vis_list, model_list, vis_slices=vis_slices,
                                context=context, facets=facets, dopsf=True, kernel=kernel)
    start = time.time()
    print("****** Starting PSF calculation ******")
    psf, sumwt = arlexecute.compute(psf_list, sync=True)[0]
    end = time.time()
    results['time psf invert'] = end - start
    print("PSF invert took %.2f seconds" % (end - start))
    
    results['psf_max'] = qa_image(psf).data['max']
    results['psf_min'] = qa_image(psf).data['min']
    
    dirty_list = invert_component(vis_list, model_list, vis_slices=vis_slices,
                                  context=context, facets=facets, kernel=kernel)
    start = time.time()
    print("****** Starting dirty image calculation ******")
    dirty, sumwt = arlexecute.compute(dirty_list, sync=True)[0]
    end = time.time()
    results['time invert'] = end - start
    print("Dirty image invert took %.2f seconds" % (end - start))
    print("Maximum in dirty image is ", numpy.max(numpy.abs(dirty.data)), ", sumwt is ", sumwt)
    qa = qa_image(dirty)
    results['dirty_max'] = qa.data['max']
    results['dirty_min'] = qa.data['min']
    
    # Create the ICAL pipeline to run 5 major cycles, starting selfcal at cycle 1. A global solution across all
    # frequencies (i.e. Visibilities) is performed.
    start = time.time()
    print("****** Starting ICAL ******")
    start = time.time()
    ical_list = ical_component(vis_list, model_imagelist=model_list, context=context, do_selfcal=1,
                               nchan=nfreqwin, vis_slices=vis_slices, algorithm='mmclean', nmoments=3,
                               niter=1000, fractional_threshold=0.1, scales=[0, 3, 10], threshold=0.1,
                               nmajor=5, gain=0.7, timeslice='auto', global_solution=True,
                               window_shape='quarter')
    end = time.time()
    results['time ICAL graph'] = end - start
    print("Construction of ICAL graph took %.2f seconds" % (end - start))
    
    # Execute the graph
    start = time.time()
    result = arlexecute.compute(ical_list, sync=True)
    deconvolved, residual, restored = result
    end = time.time()
    
    results['time ICAL'] = end - start
    print("ICAL graph execution took %.2f seconds" % (end - start))
    qa = qa_image(deconvolved[0])
    results['deconvolved_max'] = qa.data['max']
    results['deconvolved_min'] = qa.data['min']
    export_image_to_fits(deconvolved[0], "pipelines-timings-delayed-ical_deconvolved.fits")
    
    qa = qa_image(residual[0][0])
    results['residual_max'] = qa.data['max']
    results['residual_min'] = qa.data['min']
    export_image_to_fits(residual[0][0], "pipelines-timings-delayed-ical_residual.fits")
    
    qa = qa_image(restored[0])
    results['restored_max'] = qa.data['max']
    results['restored_min'] = qa.data['min']
    export_image_to_fits(restored[0], "pipelines-timings-delayed-ical_restored.fits")
    #
    arlexecute.close()
    
    end_all = time.time()
    results['time overall'] = end_all - start_all
    
    print("At end, results are {0!r}".format(results))
    
    return results


def write_results(filename, fieldnames, results):
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writerow(results)
        csvfile.close()


def write_header(filename, fieldnames):
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        csvfile.close()


def main(args):
    results = {}
    
    nworkers = args.nworkers
    results['nworkers'] = nworkers
    
    context = args.context
    results['context'] = context
    
    nnodes = args.nnodes
    results['nnodes'] = nnodes
    
    threads_per_worker = args.nthreads
    
    print("Using %s workers" % nworkers)
    print("Using %s threads per worker" % threads_per_worker)
    
    nfreqwin = args.nfreqwin
    results['nfreqwin'] = nfreqwin
    
    rmax = args.rmax
    results['rmax'] = rmax
    
    context = args.context
    results['context'] = context
    
    ntimes = args.ntimes
    results['ntimes'] = ntimes
    
    results['hostname'] = socket.gethostname()
    results['epoch'] = time.strftime("%Y-%m-%d %H:%M:%S")
    results['driver'] = 'pipelines-timings-delayed'
    
    threads_per_worker = args.nthreads
    
    print("Trying %s workers" % nworkers)
    print("Using %s threads per worker" % threads_per_worker)
    print("Defining %d frequency windows" % nfreqwin)
    
    fieldnames = ['driver', 'nnodes', 'nworkers', 'time ICAL', 'time ICAL graph', 'time create gleam',
                  'time predict', 'time corrupt', 'time invert', 'time psf invert', 'time overall',
                  'threads_per_worker', 'processes', 'order',
                  'nfreqwin', 'ntimes', 'rmax', 'facets', 'wprojection_planes', 'vis_slices', 'npixel',
                  'cellsize', 'seed', 'dirty_max', 'dirty_min', 'psf_max', 'psf_min', 'deconvolved_max',
                  'deconvolved_min', 'restored_min', 'restored_max', 'residual_max', 'residual_min',
                  'hostname', 'git_hash', 'epoch', 'context', 'use_dask']
    
    filename = seqfile.findNextFile(prefix='%s_%s_' % (results['driver'], results['hostname']), suffix='.csv')
    print('Saving results to %s' % filename)
    
    write_header(filename, fieldnames)
    
    results = trial_case(results, nworkers=nworkers, rmax=rmax, context=context,
                         threads_per_worker=threads_per_worker, nfreqwin=nfreqwin, ntimes=ntimes)
    write_results(filename, fieldnames, results)
    
    print('Exiting %s' % results['driver'])


if __name__ == '__main__':
    import csv
    import seqfile
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark pipelines in numpy and dask')
    parser.add_argument('--use_dask', type=bool, default=True, help='Use Dask?')
    parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of threads')
    parser.add_argument('--nworkers', type=int, default=1, help='Number of workers')
    
    parser.add_argument('--ntimes', type=int, default=7, help='Number of hour angles')
    parser.add_argument('--nfreqwin', type=int, default=16, help='Number of frequency windows')
    parser.add_argument('--context', type=str, default='wstack',
                        help='Imaging context: 2d|timeslice|timeslice|wstack')
    parser.add_argument('--rmax', type=float, default=300.0, help='Maximum baseline (m)')
    
    main(parser.parse_args())
    
    exit()
