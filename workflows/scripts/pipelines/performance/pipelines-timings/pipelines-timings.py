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

from data_models.polarisation import PolarisationFrame
from wrappers.arlexecute.calibration.calibration_control import create_calibration_controls
from wrappers.arlexecute.griddata.convolution_functions import convert_convolutionfunction_to_image
from wrappers.arlexecute.griddata.kernels import create_awterm_convolutionfunction
from wrappers.arlexecute.image.operations import export_image_to_fits, qa_image
from wrappers.arlexecute.imaging.base import create_image_from_visibility, advise_wide_field, \
    predict_skycomponent_visibility
from wrappers.arlexecute.imaging.primary_beams import create_pb
from wrappers.arlexecute.simulation.testing_support import create_low_test_skycomponents_from_gleam
from wrappers.arlexecute.skycomponent.operations import apply_beam_to_skycomponent, insert_skycomponent
from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility
from workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow, \
    restore_list_arlexecute_workflow
from workflows.arlexecute.pipelines.pipeline_arlexecute import ical_list_arlexecute_workflow
from workflows.arlexecute.simulation.simulation_arlexecute import simulate_list_arlexecute_workflow, \
    corrupt_list_arlexecute_workflow
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.execution_support.dask_init import findNodes, get_dask_Client

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


def trial_case(results, seed=180555, context='wstack', nworkers=8, threads_per_worker=1, memory=8,
               processes=True, order='frequency', nfreqwin=7, ntimes=3, rmax=750.0,
               facets=1, wprojection_planes=1, use_dask=True, use_serial_imaging=False,
               flux_limit=0.3):
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
    :param order: See simulate_list_list_arlexecute_workflow_workflowkflow
    :param nfreqwin: See simulate_list_list_arlexecute_workflow_workflowkflow
    :param ntimes: See simulate_list_list_arlexecute_workflow_workflowkflow
    :param rmax: See simulate_list_list_arlexecute_workflow_workflowkflow
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
    
    print("Context is %s" % context)
    
    results['nworkers'] = nworkers
    results['threads_per_worker'] = threads_per_worker
    results['processes'] = processes
    results['memory'] = memory
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
    centre = nfreqwin // 2
    if nfreqwin > 1:
        channel_bandwidth = numpy.array(nfreqwin * [frequency[1] - frequency[0]])
    else:
        channel_bandwidth = numpy.array([1e6])
    times = numpy.linspace(-numpy.pi / 3.0, numpy.pi / 3.0, ntimes)
    
    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
    
    if use_dask:
        client = get_dask_Client(threads_per_worker=threads_per_worker,
                                 memory_limit=memory * 1024 * 1024 * 1024,
                                 n_workers=nworkers)
        arlexecute.set_client(client)
        nodes = findNodes(arlexecute.client)
        print("Defined %d workers on %d nodes" % (nworkers, len(nodes)))
        print("Workers are: %s" % str(nodes))
    else:
        arlexecute.set_client(use_dask=use_dask)
        results['nnodes'] = 1
    
    vis_list = simulate_list_arlexecute_workflow('LOWBD2',
                                                 frequency=frequency,
                                                 channel_bandwidth=channel_bandwidth,
                                                 times=times,
                                                 phasecentre=phasecentre,
                                                 order=order,
                                                 format='blockvis',
                                                 rmax=rmax)
    
    print("****** Visibility creation ******")
    vis_list = arlexecute.persist(vis_list)
    
    # Find the best imaging parameters but don't bring the vis_list back here
    def get_wf(bv):
        v = convert_blockvisibility_to_visibility(bv)
        return advise_wide_field(v, guard_band_image=6.0,
                                 delA=0.02,
                                 facets=facets,
                                 wprojection_planes=wprojection_planes,
                                 oversampling_synthesised_beam=3.0)
    
    advice = arlexecute.compute(arlexecute.execute(get_wf)(vis_list[-1]), sync=True)
    
    npixel = advice['npixels2']
    cellsize = advice['cellsize']
    
    # Create an empty model image
    model_list = [arlexecute.execute(create_image_from_visibility)
                  (vis_list[f],
                   npixel=npixel, cellsize=cellsize,
                   frequency=[frequency[f]],
                   channel_bandwidth=[channel_bandwidth[f]],
                   polarisation_frame=PolarisationFrame("stokesI"))
                  for f, freq in enumerate(frequency)]
    model_list = arlexecute.compute(model_list, sync=True)
    model_list = arlexecute.scatter(model_list)
    
    def make_gleam_sc(f, v, m):
        sc = create_low_test_skycomponents_from_gleam(frequency=[frequency[f]],
                                                      phasecentre=phasecentre,
                                                      polarisation_frame=PolarisationFrame("stokesI"),
                                                      flux_limit=flux_limit)
        pb = create_pb(m, 'LOW')
        sc = apply_beam_to_skycomponent(sc, pb, flux_limit=flux_limit/100.0)
        m = insert_skycomponent(m, sc)
        return predict_skycomponent_visibility(v, sc)
    
    vis_list = [arlexecute.execute(make_gleam_sc)(f, vis_list[f], model_list[f]) for f, freq in enumerate(frequency)]
    
    start = time.time()
    print("****** Starting GLEAM visibility prediction ******")
    vis_list = arlexecute.compute(vis_list, sync=True)
    end = time.time()
    results['time create gleam'] = end - start
    print("Predicting GLEAM visibility took %.2f seconds" % (end - start))
    vis_list = arlexecute.scatter(vis_list)
    
    gcfcf_list = None
    if context == 'timeslice':
        vis_slices = ntimes
        print("Using timeslice with %d slices" % vis_slices)
    elif context == '2d':
        vis_slices = 1
    elif context == "wprojection":
        wstep = advice['wstep']
        nw = advice['wprojection_planes']
        vis_slices = 1
        support = advice['nwpixels']
        results['wprojection_planes'] = nw
        print("Using wprojection with %d planes with wstep %.1f wavelengths" % (nw, wstep))
        
        def make_gcfcf(m):
            gcf, cf = create_awterm_convolutionfunction(m, nw=nw, wstep=wstep, oversampling=8,
                                                        support=support, use_aaf=True)
            return (gcf, cf)
        
        start = time.time()
        print("****** Starting W projection kernel creation ******")
        gcfcf_list = [arlexecute.execute(make_gcfcf, nout=1)(m) for m in model_list]
        gcfcf_list = arlexecute.compute(gcfcf_list, sync=True)
        end = time.time()
        results['time create wprojection'] = end - start
        print("Creating W projection kernel took %.2f seconds" % (end - start))
        cf_image = convert_convolutionfunction_to_image(gcfcf_list[centre][1])
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "pipelines-timings-wterm-cf.fits")
        
        gcfcf_list = arlexecute.scatter(gcfcf_list)
    
    else:
        context = 'wstack'
        vis_slices = advice['vis_slices']
        print("Using wstack with %d slices" % vis_slices)
        
    psf_list = invert_list_arlexecute_workflow(vis_list, model_list, vis_slices=vis_slices, dopsf=True,
                                                 context=context, facets=facets, do_weighting=True,
                                                 use_serial_invert=use_serial_imaging, gcfcf=gcfcf_list)
    start = time.time()
    print("****** Starting psf image calculation ******")
    psf, sumwt = arlexecute.compute(psf_list, sync=True)[centre]
    end = time.time()
    results['time psf invert'] = end - start
    print("PSF invert took %.2f seconds" % (end - start))
    print("Maximum in psf image is ", numpy.max(numpy.abs(psf.data)), ", sumwt is ", sumwt)
    qa = qa_image(psf)
    results['psf_max'] = qa.data['max']
    results['psf_min'] = qa.data['min']
    export_image_to_fits(psf, "pipelines-timings-%s-psf.fits" % context)

    
    smoothed_model_list = restore_list_arlexecute_workflow(model_list, psf_list)
    smoothed_model_list = arlexecute.compute(smoothed_model_list, sync=True)
    export_image_to_fits(smoothed_model_list[centre], "pipelines-timings-cmodel.fits")
    
    # Create an empty model image
    model_list = [arlexecute.execute(create_image_from_visibility)
                  (vis_list[f],
                   npixel=npixel, cellsize=cellsize,
                   frequency=[frequency[f]],
                   channel_bandwidth=[channel_bandwidth[f]],
                   polarisation_frame=PolarisationFrame("stokesI"))
                  for f, freq in enumerate(frequency)]
    model_list = arlexecute.compute(model_list, sync=True)
    model_list = arlexecute.scatter(model_list)

    dirty_list = invert_list_arlexecute_workflow(vis_list, model_list, vis_slices=vis_slices,
                                                 context=context, facets=facets, do_weighting=True,
                                                 use_serial_invert=use_serial_imaging, gcfcf=gcfcf_list)
    start = time.time()
    print("****** Starting dirty image calculation ******")
    dirty, sumwt = arlexecute.compute(dirty_list, sync=True)[centre]
    end = time.time()
    results['time invert'] = end - start
    print("Dirty image invert took %.2f seconds" % (end - start))
    print("Maximum in dirty image is ", numpy.max(numpy.abs(dirty.data)), ", sumwt is ", sumwt)
    qa = qa_image(dirty)
    results['dirty_max'] = qa.data['max']
    results['dirty_min'] = qa.data['min']
    export_image_to_fits(dirty, "pipelines-timings-%s-dirty.fits" % context)

    # Corrupt the visibility for the GLEAM model
    print("****** Visibility corruption ******")
    vis_list = corrupt_list_arlexecute_workflow(vis_list, phase_error=1.0, seed=seed)
    start = time.time()
    vis_list = arlexecute.compute(vis_list, sync=True)
    vis_list = arlexecute.scatter(vis_list)

    end = time.time()
    results['time corrupt'] = end - start
    print("Visibility corruption took %.2f seconds" % (end - start))

    # Create the ICAL pipeline to run 5 major cycles, starting selfcal at cycle 1. A global solution across all
    # frequencies (i.e. Visibilities) is performed.
    start = time.time()
    print("****** Starting ICAL ******")
    
    controls = create_calibration_controls()
    
    controls['T']['first_selfcal'] = 1
    controls['G']['first_selfcal'] = 3
    controls['B']['first_selfcal'] = 4
    
    controls['T']['timescale'] = 'auto'
    controls['G']['timescale'] = 'auto'
    controls['B']['timescale'] = 1e5
    
    if nfreqwin > 6:
        nmoment = 3
        algorithm = 'mmclean'
    elif nfreqwin > 2:
        nmoment = 2
        algorithm = 'mmclean'
    else:
        nmoment = 1
        algorithm = 'msclean'
    
    start = time.time()
    ical_list = ical_list_arlexecute_workflow(vis_list,
                                              model_imagelist=model_list,
                                              context=context,
                                              calibration_context='TG',
                                              controls=controls,
                                              scales=[0, 3, 10],
                                              algorithm=algorithm,
                                              nmoment=nmoment,
                                              niter=1000,
                                              fractional_threshold=0.1,
                                              threshold=0.1, nmajor=5, gain=0.25,
                                              vis_slices=vis_slices,
                                              timeslice='auto',
                                              global_solution=False,
                                              psf_support=64,
                                              deconvolve_facets=8,
                                              deconvolve_overlap=32,
                                              deconvolve_taper='tukey',
                                              do_selfcal=True,
                                              use_serial_predict=use_serial_imaging,
                                              use_serial_invert=use_serial_imaging,
                                              gcfcf=gcfcf_list)
    
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
    qa = qa_image(deconvolved[centre])
    results['deconvolved_max'] = qa.data['max']
    results['deconvolved_min'] = qa.data['min']
    export_image_to_fits(deconvolved[centre], "pipelines-timings-%s-ical_deconvolved.fits" % context)
    
    qa = qa_image(residual[centre][0])
    results['residual_max'] = qa.data['max']
    results['residual_min'] = qa.data['min']
    export_image_to_fits(residual[centre][0], "pipelines-timings-%s-ical_residual.fits" % context)
    
    qa = qa_image(restored[centre])
    results['restored_max'] = qa.data['max']
    results['restored_min'] = qa.data['min']
    export_image_to_fits(restored[centre], "pipelines-timings-%s-ical_restored.fits" % context)
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
    
    results['jobid'] = args.jobid
    
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
    
    flux_limit = args.flux_limit
    results['flux_limit'] = flux_limit
    
    context = args.context
    results['context'] = context
    
    memory = args.memory
    results['memory'] = memory
    
    ntimes = args.ntimes
    results['ntimes'] = ntimes
    
    results['hostname'] = socket.gethostname()
    results['epoch'] = time.strftime("%Y-%m-%d %H:%M:%S")
    results['driver'] = 'pipelines-timings-arlexecute'
    
    use_dask = args.use_dask == 'True'
    if use_dask:
        use_serial_imaging = args.use_serial_imaging == 'True'
        print("Using Dask")
    else:
        use_serial_imaging = False
    results['use_serial_imaging'] = use_serial_imaging
    if use_serial_imaging:
        print("Using serial imaging")
    else:
        print("Using arlexecute imaging")
    
    threads_per_worker = args.nthreads
    
    print("Defining %d frequency windows" % nfreqwin)
    
    fieldnames = ['driver', 'nnodes', 'nworkers', 'time ICAL', 'time ICAL graph', 'time create gleam',
                  'time predict', 'time corrupt', 'time invert', 'time psf invert', 'time overall',
                  'threads_per_worker', 'processes', 'order',
                  'nfreqwin', 'ntimes', 'rmax', 'facets', 'wprojection_planes', 'vis_slices', 'npixel',
                  'cellsize', 'seed', 'dirty_max', 'dirty_min', 'psf_max', 'psf_min', 'deconvolved_max',
                  'deconvolved_min', 'restored_min', 'restored_max', 'residual_max', 'residual_min',
                  'hostname', 'git_hash', 'epoch', 'context', 'use_dask', 'memory', 'jobid', 'use_serial_imaging',
                  'time create wprojection', 'flux_limit']
    
    filename = seqfile.findNextFile(prefix='%s_%s_' % (results['driver'], results['hostname']), suffix='.csv')
    print('Saving results to %s' % filename)
    
    write_header(filename, fieldnames)
    
    results = trial_case(results, use_dask=use_dask, nworkers=nworkers, rmax=rmax, context=context, memory=memory,
                         threads_per_worker=threads_per_worker, nfreqwin=nfreqwin, ntimes=ntimes,
                         use_serial_imaging=use_serial_imaging, flux_limit=flux_limit)
    write_results(filename, fieldnames, results)
    
    print('Exiting %s' % results['driver'])


if __name__ == '__main__':
    import csv
    import seqfile
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark pipelines in numpy and dask')
    parser.add_argument('--use_dask', type=str, default='True', help='Use Dask?')
    parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of threads')
    parser.add_argument('--memory', type=int, default=8, help='Memory per worker')
    parser.add_argument('--nworkers', type=int, default=1, help='Number of workers')
    
    parser.add_argument('--ntimes', type=int, default=7, help='Number of hour angles')
    parser.add_argument('--nfreqwin', type=int, default=16, help='Number of frequency windows')
    parser.add_argument('--context', type=str, default='wstack',
                        help='Imaging context: 2d|timeslice|wstack')
    parser.add_argument('--rmax', type=float, default=750.0, help='Maximum baseline (m)')
    parser.add_argument('--use_serial_imaging', type=str, default='False',
                        help='Use serial imaging?')
    parser.add_argument('--jobid', type=int, default=0, help='JOBID from slurm')
    parser.add_argument('--flux_limit', type=float, default=0.3, help='Flux limit for components')

    main(parser.parse_args())
    
    exit()
