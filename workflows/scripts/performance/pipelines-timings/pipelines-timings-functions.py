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

from libs.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility
from data_models.polarisation import PolarisationFrame
from libs.image.operations import qa_image, create_empty_image_like
from libs.imaging import advise_wide_field
from libs.imaging.imaging_functions import predict_function, invert_function
from pipelines import ical
from libs.util.testing_support import create_low_test_image_from_gleam, create_blockvisibility, simulate_gaintable, \
    create_named_configuration
from libs.visibility.coalesce import convert_blockvisibility_to_visibility, convert_visibility_to_blockvisibility

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


def trial_case(results, seed=180555, context='wstack',
               order='frequency', nfreqwin=7, ntimes=3, rmax=750.0,
               facets=1, wprojection_planes=1, **kwargs):
    """ Single trial for performance-timings
    
    Simulates visibilities from GLEAM including phase errors
    Makes dirty image and PSF
    Runs ICAL pipeline
    
    The results are in a dictionary:
    
    'context': input - a string describing concisely the purpose of the test
    'time overall',  overall execution time (s)
    'time create gleam', time to create GLEAM model
    'time predict', time to execute GLEAM prediction graph
    'time corrupt', time to corrupt data_models
    'time invert', time to make dirty image
    'time psf invert', time to make PSF
    'time ICAL', time to execute ICAL
    'context', type of imaging e.g. 'wstack'
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
    :param context: Type of context: '2d'|'timeslice'|'timeslice_single'|'wstack'|'wstack_single'
    :param order: See create_simulate_vis_graph
    :param nfreqwin: See create_simulate_vis_graph
    :param ntimes: See create_simulate_vis_graph
    :param rmax: See create_simulate_vis_graph
    :param facets: Number of facets to use
    :param wprojection_planes: Number of wprojection planes to use
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
    
    print("****** Visibility creation ******")
    config = create_named_configuration('LOWBD2', rmax=rmax)
    block_vis = create_blockvisibility(config,
                                       frequency=frequency,
                                       channel_bandwidth=channel_bandwidth,
                                       times=times,
                                       phasecentre=phasecentre,
                                       polarisation_frame=PolarisationFrame("stokesI"))
    
    # Find the best imaging parameters.
    wprojection_planes = 1
    advice = advise_wide_field(block_vis, guard_band_image=6.0, delA=0.02, facets=facets,
                               wprojection_planes=wprojection_planes, oversampling_synthesised_beam=4.0)
    
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
    
    # Create a realistic image using GLEAM and apply the primary beam.
    start = time.time()
    print("****** Starting GLEAM model creation ******")
    gleam_model = create_low_test_image_from_gleam(npixel=npixel,
                                                   frequency=frequency,
                                                   channel_bandwidth=channel_bandwidth,
                                                   cellsize=cellsize,
                                                   phasecentre=phasecentre,
                                                   polarisation_frame=PolarisationFrame("stokesI"),
                                                   applybeam=True, flux_limit=0.1)
    end = time.time()
    results['time create gleam'] = end - start
    print("Creating GLEAM model took %.2f seconds" % (end - start))
    empty_model = create_empty_image_like(gleam_model)
    
    start = time.time()
    print("****** Starting GLEAM model visibility prediction ******")
    vis = convert_blockvisibility_to_visibility(block_vis)
    vis = predict_function(vis, gleam_model, vis_slices=vis_slices, context=context)
    end = time.time()
    results['time predict'] = end - start
    print("GLEAM model Visibility prediction took %.2f seconds" % (end - start))
    
    # Now we need to convert to block format for the corruption
    block_vis = convert_visibility_to_blockvisibility(vis)
    
    print("****** Visibility corruption ******")
    start = time.time()
    gt = create_gaintable_from_blockvisibility(block_vis)
    gt = simulate_gaintable(gt, phase_error=1.0)
    block_vis = apply_gaintable(block_vis, gt)
    end = time.time()
    results['time corrupt'] = end - start
    print("Visibility corruption took %.2f seconds" % (end - start))
    
    vis = convert_blockvisibility_to_visibility(block_vis)
    print("****** Starting PSF calculation ******")
    start = time.time()
    psf, sumwt = invert_function(vis, empty_model, vis_slices=vis_slices, facets=facets, dopsf=True,
                                 context=context)
    end = time.time()
    results['time psf invert'] = end - start
    print("PSF invert took %.2f seconds" % (end - start))
    results['psf_max'] = qa_image(psf).data['max']
    results['psf_min'] = qa_image(psf).data['min']
    
    print("****** Starting dirty image calculation ******")
    start = time.time()
    dirty, sumwt = invert_function(vis, empty_model, vis_slices=vis_slices, facets=facets,
                                   dopsf=False, context=context, **kwargs)
    end = time.time()
    results['time invert'] = end - start
    print("Dirty image invert took %.2f seconds" % (end - start))
    print("Maximum in dirty image is ", numpy.max(numpy.abs(dirty.data)), ", sumwt is ", sumwt)
    qa = qa_image(dirty)
    results['dirty_max'] = qa.data['max']
    results['dirty_min'] = qa.data['min']
    
    # Create the ICAL pipeline to run 5 major cycles, starting selfcal at cycle 1. A global solution across all
    # frequencies (i.e. Visibilities) is performed.
    print("****** Starting ICAL ******")
    start = time.time()
    result = ical(block_vis,
                  model=empty_model,
                  context=context,
                  vis_slices=vis_slices,
                  algorithm='mmclean', niter=1000,
                  nmoments=3, scales=[0, 3, 10, 30],
                  fractional_threshold=0.1,
                  threshold=0.01, nmajor=5, gain=0.7,
                  first_selfcal=1, timeslice='auto',
                  global_solution=False,
                  kernel=kernel)
    
    end = time.time()
    
    deconvolved, residual, restored = result
    
    results['time ICAL'] = end - start
    print("ICAL compute took %.2f seconds" % (end - start))
    qa = qa_image(deconvolved)
    results['deconvolved_max'] = qa.data['max']
    results['deconvolved_min'] = qa.data['min']
    qa = qa_image(residual)
    results['residual_max'] = qa.data['max']
    results['residual_min'] = qa.data['min']
    qa = qa_image(restored)
    results['restored_max'] = qa.data['max']
    results['restored_min'] = qa.data['min']
    end_all = time.time()
    
    results['time overall'] = end_all - start_all
    
    print("At end, results are {0!r}".format(results))
    
    from libs.image.operations import export_image_to_fits
    export_image_to_fits(deconvolved, "ical_deconvolved.fits")
    export_image_to_fits(residual, "ical_residual.fits")
    export_image_to_fits(restored, "ical_restored.fits")
    
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
    
    nfreq = args.nfreqwin
    results['nfreqwin'] = nfreq
    
    rmax = args.rmax
    results['rmax'] = rmax
    
    context = args.context
    results['context'] = context
    
    ntimes = args.ntimes
    results['ntimes'] = ntimes
    
    results['hostname'] = socket.gethostname()
    results['epoch'] = time.strftime("%Y-%m-%d %H:%M:%S")
    results['driver'] = 'pipelines-timings-functions'
    
    print("Defining %d frequency windows" % nfreq)
    
    fieldnames = ['driver', 'nnodes', 'nworkers', 'time ICAL', 'time ICAL graph', 'time create gleam',
                  'time predict', 'time corrupt', 'time invert', 'time psf invert', 'time overall',
                  'threads_per_worker', 'processes', 'order',
                  'nfreqwin', 'ntimes', 'rmax', 'facets', 'wprojection_planes', 'vis_slices', 'npixel',
                  'cellsize', 'seed', 'dirty_max', 'dirty_min', 'psf_max', 'psf_min', 'deconvolved_max',
                  'deconvolved_min', 'restored_min', 'restored_max', 'residual_max', 'residual_min',
                  'hostname', 'git_hash', 'epoch', 'context']
    
    filename = seqfile.findNextFile(prefix='%s_%s_' % (results['driver'], results['hostname']), suffix='.csv')
    print('Saving results to %s' % filename)
    
    write_header(filename, fieldnames)
    
    results = trial_case(results, context=context, rmax=rmax, nfreqwin=nfreq, ntimes=ntimes)
    write_results(filename, fieldnames, results)
    
    print('Exiting %s' % results['driver'])


if __name__ == '__main__':
    import csv
    import seqfile
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark pipelines in numpy')
    
    parser.add_argument('--ntimes', type=int, default=7, help='Number of hour angles')
    parser.add_argument('--nfreqwin', type=int, default=16, help='Number of frequency windows')
    parser.add_argument('--context', type=str, default='timeslice_single',
                        help='Imaging context: 2d|timeslice|timeslice_single|wstack|wstack_single')
    parser.add_argument('--rmax', type=float, default=300.0, help='Maximum baseline (m)')
    
    main(parser.parse_args())
    
    exit()
