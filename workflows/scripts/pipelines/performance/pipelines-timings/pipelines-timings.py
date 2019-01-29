# # Pipeline processing using Dask
#
import logging
import socket
import time

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow, \
    weight_list_arlexecute_workflow, \
    taper_list_arlexecute_workflow, remove_sumwt
from workflows.arlexecute.pipelines.pipeline_arlexecute import ical_list_arlexecute_workflow
from workflows.arlexecute.simulation.simulation_arlexecute import simulate_list_arlexecute_workflow, \
    corrupt_list_arlexecute_workflow
from workflows.arlexecute.skymodel.skymodel_arlexecute import predict_skymodel_list_arlexecute_workflow
from wrappers.arlexecute.calibration.calibration_control import create_calibration_controls
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.execution_support.dask_init import findNodes, get_dask_Client
from wrappers.arlexecute.griddata.convolution_functions import convert_convolutionfunction_to_image
from wrappers.arlexecute.griddata.kernels import create_awterm_convolutionfunction
from wrappers.arlexecute.image.gather_scatter import image_gather_channels
from wrappers.arlexecute.image.operations import export_image_to_fits, qa_image
from wrappers.arlexecute.imaging.base import create_image_from_visibility, advise_wide_field
from wrappers.arlexecute.simulation.testing_support import create_low_test_skymodel_from_gleam
from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility

import pprint

pp = pprint.PrettyPrinter()


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
               flux_limit=0.3, nmajor=5, dft_threshold=1.0):
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
    if use_dask:
        client = get_dask_Client(threads_per_worker=threads_per_worker,
                                 processes = threads_per_worker == 1,
                                 memory_limit=memory * 1024 * 1024 * 1024,
                                 n_workers=nworkers)
        arlexecute.set_client(client)
        nodes = findNodes(arlexecute.client)
        print("Defined %d workers on %d nodes" % (nworkers, len(nodes)))
        print("Workers are: %s" % str(nodes))
    else:
        arlexecute.set_client(use_dask=use_dask)
        results['nnodes'] = 1

    def init_logging():
        logging.basicConfig(filename='pipelines-timings.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

    init_logging()
    log = logging.getLogger()
    
    # Initialise logging on the workers. This appears to only work using the process scheduler.
    arlexecute.run(init_logging)

    
    def lprint(s):
        log.info(s)
        print(s)
    
    lprint("Starting pipelines-timings")
    
    numpy.random.seed(seed)
    results['seed'] = seed
    
    start_all = time.time()
    
    results['context'] = context
    results['hostname'] = socket.gethostname()
    results['git_hash'] = git_hash()
    results['epoch'] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    lprint("Context is %s" % context)
    
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
    results['dft threshold'] = dft_threshold
    
    results['use_dask'] = use_dask
    
    lprint("At start, configuration is:")
    lprint(results)
    
    # Parameters determining scale
    frequency = numpy.linspace(1.0e8, 1.2e8, nfreqwin)
    centre = nfreqwin // 2
    if nfreqwin > 1:
        channel_bandwidth = numpy.array(nfreqwin * [frequency[1] - frequency[0]])
    else:
        channel_bandwidth = numpy.array([1e6])
    
    times = numpy.linspace(-numpy.pi / 4.0, numpy.pi / 4.0, ntimes)
    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
    
    bvis_list = simulate_list_arlexecute_workflow('LOWBD2',
                                                  frequency=frequency,
                                                  channel_bandwidth=channel_bandwidth,
                                                  times=times,
                                                  phasecentre=phasecentre,
                                                  order=order,
                                                  format='blockvis',
                                                  rmax=rmax)
    
    lprint("****** Visibility creation ******")
    bvis_list = arlexecute.compute(bvis_list, sync=True)
    
    vis_list = [arlexecute.execute(convert_blockvisibility_to_visibility(bv)) for bv in bvis_list]
    vis_list = arlexecute.compute(vis_list, sync=True)
    
    # Find the best imaging parameters but don't bring the vis_list back here
    def get_wf(v):
        return advise_wide_field(v, guard_band_image=6.0,
                                 delA=0.1,
                                 facets=facets,
                                 wprojection_planes=wprojection_planes,
                                 oversampling_synthesised_beam=4.0)
    
    advice = arlexecute.compute(arlexecute.execute(get_wf)(vis_list[-1]), sync=True)
    
    # Deconvolution via sub-images requires 2^n
    npixel = advice['npixels2']
    results['npixel'] = npixel
    cellsize = advice['cellsize']
    results['cellsize'] = cellsize
    lprint("Image will have %d by %d pixels, cellsize = %.6f rad" % (npixel, npixel, cellsize))
    
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
    
    start = time.time()
    vis_list = weight_list_arlexecute_workflow(vis_list, model_list)
    vis_list = taper_list_arlexecute_workflow(vis_list, 0.003 * 750.0 / rmax)
    print("****** Starting weighting and tapering ******")
    vis_list = arlexecute.compute(vis_list, sync=True)
    end = time.time()
    results['time weight'] = end - start
    print("Weighting took %.3f seconds" % (end - start))
    vis_list = arlexecute.scatter(vis_list)
    
    # Now set up the imaging parameters
    gcfcf_list = None
    if context == 'timeslice':
        vis_slices = ntimes
        lprint("Using timeslice with %d slices" % vis_slices)
    elif context == '2d':
        vis_slices = 1
    elif context == "wprojection":
        wstep = advice['wstep']
        nw = advice['wprojection_planes']
        vis_slices = 1
        support = advice['nwpixels']
        results['wprojection_planes'] = nw
        lprint("Using wprojection with %d planes with wstep %.1f wavelengths" % (nw, wstep))
        
        start = time.time()
        lprint("****** Starting W projection kernel creation ******")
        gcfcf_list = [arlexecute.execute(create_awterm_convolutionfunction, nout=1)
                      (m, nw=nw, wstep=wstep, oversampling=8, support=support, use_aaf=True)
                      for m in model_list]
        gcfcf_list = arlexecute.compute(gcfcf_list, sync=True)
        end = time.time()
        results['time create wprojection'] = end - start
        lprint("Creating W projection kernel took %.3f seconds" % (end - start))
        cf_image = convert_convolutionfunction_to_image(gcfcf_list[centre][1])
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "pipelines-timings-wterm-cf.fits")
        
        gcfcf_list = arlexecute.scatter(gcfcf_list)
    
    else:
        context = 'wstack'
        vis_slices = advice['vis_slices']
        lprint("Using wstack with %d slices" % vis_slices)
    
    results['vis_slices'] = vis_slices
    
    # Make a skymodel from gleam, with bright sources as components and weak sources in an image
    lprint("****** Starting GLEAM skymodel creation ******")
    start = time.time()
    skymodel_list = [arlexecute.execute(create_low_test_skymodel_from_gleam)
                     (npixel=npixel, cellsize=cellsize, frequency=[frequency[f]],
                      phasecentre=phasecentre,
                      polarisation_frame=PolarisationFrame("stokesI"),
                      flux_limit=flux_limit,
                      flux_threshold=dft_threshold,
                      flux_max=5.0) for f, freq in enumerate(frequency)]
    skymodel_list = arlexecute.compute(skymodel_list, sync=True)
    end = time.time()
    lprint("GLEAM skymodel creation took %.3f seconds" % (end - start))
    results['time create gleam'] = end - start
    
    lprint("****** Starting GLEAM skymodel prediction ******")
    start = time.time()
    predicted_vis_list = predict_skymodel_list_arlexecute_workflow(vis_list, skymodel_list, context=context,
                                                                   vis_slices=vis_slices, facets=facets,
                                                                   gcfcf=gcfcf_list)
    predicted_vis_list = arlexecute.compute(predicted_vis_list, sync=True)
    end = time.time()
    lprint("GLEAM skymodel prediction took %.3f seconds" % (end - start))
    results['time predict gleam'] = end - start
    
    lprint("****** Starting psf image calculation ******")
    start = time.time()
    predicted_vis_list = arlexecute.scatter(predicted_vis_list)
    psf_list = invert_list_arlexecute_workflow(predicted_vis_list, model_list, vis_slices=vis_slices,
                                               dopsf=True, context=context, facets=facets,
                                               use_serial_invert=use_serial_imaging, gcfcf=gcfcf_list)
    psf, sumwt = arlexecute.compute(psf_list, sync=True)[centre]
    end = time.time()
    results['time psf invert'] = end - start
    lprint("PSF invert took %.3f seconds" % (end - start))
    lprint("Maximum in psf image is %f, sumwt is %s" % (numpy.max(numpy.abs(psf.data)), str(sumwt)))
    qa = qa_image(psf)
    results['psf_max'] = qa.data['max']
    results['psf_min'] = qa.data['min']
    export_image_to_fits(psf, "pipelines-timings-%s-psf.fits" % context)
    
    # Make a smoothed model image for comparison
    
    # smoothed_model_list = restore_list_arlexecute_workflow(gleam_model_list, psf_list)
    # smoothed_model_list = arlexecute.compute(smoothed_model_list, sync=True)
    # smoothed_cube = image_gather_channels(smoothed_model_list)
    # export_image_to_fits(smoothed_cube, "pipelines-timings-cmodel.fits")
    
    # Create an empty model image
    model_list = [arlexecute.execute(create_image_from_visibility)
                  (predicted_vis_list[f],
                   npixel=npixel, cellsize=cellsize,
                   frequency=[frequency[f]],
                   channel_bandwidth=[channel_bandwidth[f]],
                   polarisation_frame=PolarisationFrame("stokesI"))
                  for f, freq in enumerate(frequency)]
    model_list = arlexecute.compute(model_list, sync=True)
    model_list = arlexecute.scatter(model_list)
    
    lprint("****** Starting dirty image calculation ******")
    start = time.time()
    dirty_list = invert_list_arlexecute_workflow(predicted_vis_list, model_list, vis_slices=vis_slices,
                                                 context=context, facets=facets,
                                                 use_serial_invert=use_serial_imaging, gcfcf=gcfcf_list)
    dirty, sumwt = arlexecute.compute(dirty_list, sync=True)[centre]
    end = time.time()
    results['time invert'] = end - start
    lprint("Dirty image invert took %.3f seconds" % (end - start))
    lprint("Maximum in dirty image is %f, sumwt is %s" % (numpy.max(numpy.abs(dirty.data)), str(sumwt)))
    qa = qa_image(dirty)
    results['dirty_max'] = qa.data['max']
    results['dirty_min'] = qa.data['min']
    export_image_to_fits(dirty, "pipelines-timings-%s-dirty.fits" % context)
    
    # Corrupt the visibility for the GLEAM model
    lprint("****** Visibility corruption ******")
    start = time.time()
    corrupted_vis_list = corrupt_list_arlexecute_workflow(predicted_vis_list, phase_error=1.0, seed=seed)
    corrupted_vis_list = arlexecute.compute(corrupted_vis_list, sync=True)
    end = time.time()
    results['time corrupt'] = end - start
    lprint("Visibility corruption took %.3f seconds" % (end - start))
    
    # Create the ICAL pipeline to run major cycles, starting selfcal at cycle 1. A global solution across all
    # frequencies (i.e. Visibilities) is performed.
    lprint("****** Starting ICAL ******")
    
    controls = create_calibration_controls()
    
    controls['T']['first_selfcal'] = 1
    controls['T']['timescale'] = 'auto'
    
    start = time.time()
    ical_list = ical_list_arlexecute_workflow(corrupted_vis_list,
                                              model_imagelist=model_list,
                                              context=context,
                                              vis_slices=vis_slices,
                                              scales=[0, 3, 10],
                                              algorithm='mmclean',
                                              nmoment=3, niter=1000,
                                              fractional_threshold=0.1,
                                              threshold=0.01, nmajor=nmajor,
                                              gain=0.25,
                                              psf_support=64,
                                              deconvolve_facets=8,
                                              deconvolve_overlap=32,
                                              deconvolve_taper='tukey',
                                              timeslice='auto',
                                              global_solution=True,
                                              do_selfcal=True,
                                              calibration_context='T',
                                              controls=controls,
                                              use_serial_predict=use_serial_imaging,
                                              use_serial_invert=use_serial_imaging,
                                              gcfcf=gcfcf_list)
    end = time.time()
    results['time ICAL graph'] = end - start
    lprint("Construction of ICAL graph took %.3f seconds" % (end - start))
    
    # Execute the graph
    start = time.time()
    result = arlexecute.compute(ical_list, sync=True)
    deconvolved, residual, restored = result
    end = time.time()
    
    results['time ICAL'] = end - start
    lprint("ICAL graph execution took %.3f seconds" % (end - start))
    qa = qa_image(deconvolved[centre])
    results['deconvolved_max'] = qa.data['max']
    results['deconvolved_min'] = qa.data['min']
    deconvolved_cube = image_gather_channels(deconvolved)
    export_image_to_fits(deconvolved_cube, "pipelines-timings-%s-ical_deconvolved.fits" % context)
    
    qa = qa_image(residual[centre][0])
    results['residual_max'] = qa.data['max']
    results['residual_min'] = qa.data['min']
    residual_cube = remove_sumwt(residual)
    residual_cube = image_gather_channels(residual_cube)
    export_image_to_fits(residual_cube, "pipelines-timings-%s-ical_residual.fits" % context)
    
    qa = qa_image(restored[centre])
    results['restored_max'] = qa.data['max']
    results['restored_min'] = qa.data['min']
    restored_cube = image_gather_channels(restored)
    export_image_to_fits(restored_cube, "pipelines-timings-%s-ical_restored.fits" % context)
    #
    arlexecute.close()
    
    end_all = time.time()
    results['time overall'] = end_all - start_all
    
    lprint("At end, results are:")
    lprint(results)
    
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
    
    dft_threshold = args.dft_threshold
    results['dft threshold'] = dft_threshold
    
    context = args.context
    results['context'] = context
    
    memory = args.memory
    results['memory'] = memory
    
    ntimes = args.ntimes
    results['ntimes'] = ntimes
    
    nmajor = args.nmajor
    results['nmajor'] = nmajor
    
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
                  'time predict gleam', 'dft threshold',
                  'time corrupt', 'time invert', 'time psf invert', 'time weight', 'time overall',
                  'threads_per_worker', 'processes', 'order',
                  'nfreqwin', 'ntimes', 'rmax', 'facets', 'wprojection_planes', 'vis_slices', 'npixel',
                  'cellsize', 'seed', 'dirty_max', 'dirty_min', 'psf_max', 'psf_min', 'deconvolved_max',
                  'deconvolved_min', 'restored_min', 'restored_max', 'residual_max', 'residual_min',
                  'hostname', 'git_hash', 'epoch', 'context', 'use_dask', 'memory', 'jobid', 'use_serial_imaging',
                  'time create wprojection', 'flux_limit', 'nmajor', 'log_file']
    
    filename = seqfile.findNextFile(prefix='%s_%s_' % (results['driver'], results['hostname']), suffix='.csv')
    print('Saving results to %s' % filename)
    
    write_header(filename, fieldnames)
    
    results = trial_case(results, use_dask=use_dask, nworkers=nworkers, rmax=rmax, context=context, memory=memory,
                         threads_per_worker=threads_per_worker, nfreqwin=nfreqwin, ntimes=ntimes,
                         use_serial_imaging=use_serial_imaging, flux_limit=flux_limit, nmajor=nmajor,
                         dft_threshold=dft_threshold)
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
    parser.add_argument('--nmajor', type=int, default=5, help='Number of major cycles')
    
    parser.add_argument('--ntimes', type=int, default=7, help='Number of hour angles')
    parser.add_argument('--nfreqwin', type=int, default=16, help='Number of frequency windows')
    parser.add_argument('--context', type=str, default='wstack',
                        help='Imaging context: 2d|timeslice|wstack')
    parser.add_argument('--rmax', type=float, default=750.0, help='Maximum baseline (m)')
    parser.add_argument('--use_serial_imaging', type=str, default='False',
                        help='Use serial imaging?')
    parser.add_argument('--jobid', type=int, default=0, help='JOBID from slurm')
    parser.add_argument('--flux_limit', type=float, default=0.3, help='Flux limit for components')
    parser.add_argument('--dft_threshold', type=float, default=1.0, help='Flux above which DFT is used')
    parser.add_argument('--log_file', type=str, default='pipelines-timings.log',
                        help='Name of output log file')
    
    main(parser.parse_args())
    
    exit()
