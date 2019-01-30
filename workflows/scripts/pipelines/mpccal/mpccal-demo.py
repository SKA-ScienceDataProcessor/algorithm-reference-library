# coding: utf-8

# ## Simulate non-isoplanatic imaging for LOW at 100MHz.
# 
# ### A set of model components are drawn from GLEAM. An ionospheric screen model is used to calculate
# the pierce points of the two stations in an interferometer for a given component. The model visibilities
# are calculated directly, and screen phase applied to obtain the corrupted visibility.
#
# To make an image, first isoplanatic selfcalibration is used.

# In[ ]:

import logging
import sys

from functools import partial

import matplotlib.pyplot as plt

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.memory_data_models import SkyModel
from data_models.polarisation import PolarisationFrame
from data_models.parameters import arl_path
from processing_library.image.operations import create_empty_image_like, copy_image
from wrappers.arlexecute.image.operations import export_image_to_fits

from wrappers.arlexecute.calibration.operations import create_gaintable_from_blockvisibility, copy_gaintable
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.execution_support.dask_init import get_dask_Client
from wrappers.arlexecute.skycomponent.operations import remove_neighbouring_components, \
    find_skycomponents
from wrappers.arlexecute.skymodel.operations import calculate_skymodel_equivalent_image
from wrappers.arlexecute.skymodel.operations import initialize_skymodel_voronoi
from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility
from wrappers.serial.image.operations import import_image_from_fits
from wrappers.serial.image.operations import qa_image, show_image
from wrappers.serial.imaging.base import create_image_from_visibility, advise_wide_field
from wrappers.serial.imaging.primary_beams import create_low_test_beam
from wrappers.serial.simulation.ionospheric_screen import create_gaintable_from_screen
from wrappers.serial.simulation.testing_support import create_named_configuration, \
    create_low_test_skycomponents_from_gleam
from wrappers.serial.skycomponent.operations import apply_beam_to_skycomponent, insert_skycomponent
from wrappers.serial.skycomponent.operations import filter_skycomponents_by_flux
from wrappers.serial.visibility.base import create_blockvisibility, copy_visibility, create_visibility_from_rows
from wrappers.serial.visibility.vis_select import vis_select_uvrange

from workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow
from workflows.arlexecute.imaging.imaging_arlexecute import restore_list_arlexecute_workflow
from workflows.arlexecute.pipelines.pipeline_mpccal_arlexecute import mpccal_skymodel_list_arlexecute_workflow
from workflows.arlexecute.skymodel.skymodel_arlexecute import predict_skymodel_list_arlexecute_workflow
from workflows.serial.imaging.imaging_serial import weight_list_serial_workflow, taper_list_serial_workflow


if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stdout))

    import argparse

    parser = argparse.ArgumentParser(description='MPCCAL pipeline example')
    
    parser.add_argument('--use_dask', type=str, default='True', help='Use Dask?')
    parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of threads')
    parser.add_argument('--memory', type=int, default=8, help='Memory per worker')
    parser.add_argument('--nworkers', type=int, default=4, help='Number of workers')
    parser.add_argument('--ical_nmajor', type=int, default=10, help='Number of major cycles for ICAL')
    parser.add_argument('--mpccal_nmajor', type=int, default=20, help='Number of major cycles for MPCCAL')
    parser.add_argument('--ntimes', type=int, default=3, help='Number of hour angles')
    parser.add_argument('--block', type=str, default='False', help='Number of hour angles')
    parser.add_argument('--context', type=str, default='2d',
                        help='Imaging context: 2d|timeslice|wstack')
    parser.add_argument('--rmax', type=float, default=2500.0, help='Maximum baseline (m)')
    parser.add_argument('--ical_components', type=int, default=0, help='Number of components to use in initial ICAL '
                                                                       'model')
    parser.add_argument('--flux_limit', type=float, default=0.2, help='Flux limit for components')
    parser.add_argument('--ical_threshold', type=float, default=100.0, help='ICAL source finding threshold in median '
                                                                            'abs '
                                                                            'dev med')
    parser.add_argument('--mpccal_threshold', type=float, default=10.0, help='MPCCAL source finding threshold in '
                                                                              'median abs '
                                                                            'dev med')
    args = parser.parse_args()

    block_plots = args.block == 'True'

    use_dask = args.use_dask
    n_workers = args.nworkers
    
    if use_dask:
        c = get_dask_Client(memory_limit=64 * 1024 * 1024 * 1024, n_workers=n_workers, threads_per_worker=1)
        arlexecute.set_client(c)
    else:
        arlexecute.set_client(use_dask=False)
    #######################################################################################################
    # Set up the observation: 10 minutes at transit, with 10s integration.
    # Skip 5/6 points to avoid out station redundancy
    
    nfreqwin = 1
    ntimes = args.ntimes
    rmax = args.rmax
    dec = -40.0 * u.deg
    frequency = [1e8]
    channel_bandwidth = [0.1e8]
    times = numpy.linspace(-10.0, 10.0, ntimes) * numpy.pi / (3600.0 * 12.0)
    
    phasecentre = SkyCoord(ra=+0.0 * u.deg, dec=dec, frame='icrs', equinox='J2000')
    low = create_named_configuration('LOWBD2', rmax=rmax)
    print('Configuration has %d stations' % len(low.data))
    centre = numpy.mean(low.xyz, axis=0)
    distance = numpy.hypot(low.xyz[:, 0] - centre[0],
                           low.xyz[:, 1] - centre[1],
                           low.xyz[:, 2] - centre[2])
    lowouter = low.data[distance > 1000.0][::6]
    lowcore = low.data[distance < 1000.0][::3]
    low.data = numpy.hstack((lowcore, lowouter))
    
    block_vis = create_blockvisibility(
        low,
        times,
        frequency=frequency,
        channel_bandwidth=channel_bandwidth,
        weight=1.0,
        phasecentre=phasecentre,
        polarisation_frame=PolarisationFrame("stokesI"),
        zerow=True)
    
    vis = convert_blockvisibility_to_visibility(block_vis)
    advice = advise_wide_field(vis, guard_band_image=2.0, delA=0.02)
    
    cellsize = advice['cellsize']
    vis_slices = advice['vis_slices']
    npixel = advice['npixels2']
    
    small_model = create_image_from_visibility(
        block_vis,
        npixel=512,
        frequency=frequency,
        nchan=nfreqwin,
        cellsize=cellsize,
        phasecentre=phasecentre)
    
    vis.data['imaging_weight'][...] = vis.data['weight'][...]
    vis = weight_list_serial_workflow([vis], [small_model])[0]
    vis = taper_list_serial_workflow([vis], 3 * cellsize)[0]
    
    block_vis = convert_visibility_to_blockvisibility(vis)
    
    #######################################################################################################
    ### Generate the model from the GLEAM catalog, including application of the primary beam.
    flux_limit = args.flux_limit
    beam = create_image_from_visibility(
        block_vis,
        npixel=npixel,
        frequency=frequency,
        nchan=nfreqwin,
        cellsize=cellsize,
        phasecentre=phasecentre)
    beam = create_low_test_beam(beam)
    
    original_gleam_components = create_low_test_skycomponents_from_gleam(
        flux_limit=flux_limit,
        phasecentre=phasecentre,
        frequency=frequency,
        polarisation_frame=PolarisationFrame('stokesI'),
        radius=0.15)
    
    all_components = apply_beam_to_skycomponent(original_gleam_components, beam)
    all_components = filter_skycomponents_by_flux(all_components, flux_min=flux_limit)
    all_components = sorted(all_components, key=lambda comp: numpy.max(comp.flux), reverse=True)
    print("Number of components in simulation %d" % len(all_components))
    
    screen = import_image_from_fits(arl_path('data/models/test_mpc_screen.fits'))
    all_gaintables = create_gaintable_from_screen(block_vis, all_components, screen)
    
    all_skymodel = [SkyModel(components=[all_components[i]], gaintable=all_gaintables[i])
                    for i, sm in enumerate(all_components)]
    all_skymodel_blockvis = copy_visibility(block_vis, zero=True)
    all_skymodel_vis = convert_blockvisibility_to_visibility(all_skymodel_blockvis)
    
    #######################################################################################################
    # Cqlculate visibility by using the predict_skymodel function.
    ngroup = n_workers
    future_vis = arlexecute.scatter(all_skymodel_vis)
    chunks = [all_skymodel[i:i + ngroup] for i in range(0, len(all_skymodel), ngroup)]
    for chunk in chunks:
        result = predict_skymodel_list_arlexecute_workflow(future_vis, chunk, context='2d', docal=True)
        work_vis = arlexecute.compute(result, sync=True)
        for w in work_vis:
            all_skymodel_vis.data['vis'] += w.data['vis']
        assert numpy.max(numpy.abs(all_skymodel_vis.data['vis'])) > 0.0
    
    all_skymodel_blockvis = convert_visibility_to_blockvisibility(all_skymodel_vis)

    model = create_image_from_visibility(block_vis, npixel=npixel, frequency=frequency, nchan=nfreqwin,
        cellsize=cellsize, phasecentre=phasecentre)

    def progress(res, tl_list, gt_list, it, context='MPCCAL'):
        print('Iteration %d' % it)
        
        print('Length of theta = %d' % len(tl_list))
        
        print(qa_image(res, context='%s residual image: iteration %d' % (context, it)))
        export_image_to_fits(res, arl_path("test_results/low-sims-mpc-%s-residual_iteration%d_rmax%.1f.fits" %
                             (context, it, rmax)))
        show_image(res, title='%s residual image: iteration %d' % (context, it))
        plt.show(block=block_plots)
        
        combined_model = calculate_skymodel_equivalent_image(tl_list)
        print(qa_image(combined_model, context='Combined model: iteration %d' % it))
        export_image_to_fits(combined_model, arl_path("test_results/low-sims-mpc-%s-model_iteration%d_rmax%.1f.fits" %
                             (context, it, rmax)))
        
        plt.clf()
        for i in range(len(tl_list)):
            plt.plot(numpy.angle(tl_list[i].gaintable.gain[:, :, 0, 0, 0]).flatten(),
                     numpy.angle(gt_list[i]['T'].gain[:, :, 0, 0, 0]).flatten(),
                     '.')
        plt.xlabel('Current phase')
        plt.ylabel('Update to phase')
        plt.title("%s iteration%d: Change in phase" % (context, it))
        plt.savefig(arl_path("test_results/low-sims-mpc-%s-skymodel-phase-change_iteration%d.jpg" %
                    (context, it)))
        plt.show(block=block_plots)
        return tl_list

    null_gaintable = copy_gaintable(all_gaintables[0])
    null_gaintable.data['gain'][...] = 1.0+0.0j
    
    future_vis = arlexecute.scatter(all_skymodel_vis)

    if args.ical_components > 0:
        initial_components = all_components[:args.ical_components]
        initial_model = create_empty_image_like(model)
        initial_model = insert_skycomponent(initial_model, initial_components)
        print("Number of components in ICAL initial model %d" % len(initial_components))
        ical_skymodel = [SkyModel(components=[all_components[0]], gaintable=all_gaintables[0],
                                  image=initial_model)]
    else:
        ical_skymodel = [SkyModel(components=[all_components[0]], gaintable=all_gaintables[0], image=model)]

    future_model = arlexecute.scatter(model)
    future_theta_list = arlexecute.scatter(ical_skymodel)
    result = mpccal_skymodel_list_arlexecute_workflow(future_vis, future_model, future_theta_list,
                                                      mpccal_progress=partial(progress, context='ICAL'),
                                                      nmajor=args.ical_nmajor,
                                                      context='2d',
                                                      algorithm='hogbom',
                                                      fractional_threshold=0.3, threshold=0.1,
                                                      gain=0.1, niter=1000,
                                                      psf_support=512,
                                                      deconvolve_facets=8,
                                                      deconvolve_overlap=16,
                                                      deconvolve_taper='tukey')
    
    (ical_skymodel, residual) = arlexecute.compute(result, sync=True)
    print(qa_image(residual, context='ICAL residual image'))
    
    print('mpccal finished')
    
    combined_model = calculate_skymodel_equivalent_image(ical_skymodel)
    print(qa_image(combined_model, context='ICAL combined model'))
    psf_obs = invert_list_arlexecute_workflow([future_vis], [future_model], context='2d', dopsf=True)
    result = restore_list_arlexecute_workflow([combined_model], psf_obs, [(residual, 0.0)])
    result = arlexecute.compute(result, sync=True)
    ical_restored = result[0]
    
    export_image_to_fits(ical_restored, arl_path('test_results/low-sims-mpc-ical-restored_%.1frmax.fits' % rmax))
    
    ical_finding_threshold = args.ical_threshold * qa_image(ical_restored).data['medianabs']
    ical_components = find_skycomponents(ical_restored, fwhm=2,
                                         threshold=ical_finding_threshold, npixels=12)
    # ### Remove weaker of components that are too close (0.02 rad)
    idx, ical_components = remove_neighbouring_components(ical_components, 0.02)
    print("Voronoi decomposition based on %d point sources" % len(ical_components))
    
    print(qa_image(ical_restored, context='ICAL restored image'))
    show_image(ical_restored, title='ICAL restored image', components=ical_components)
    plt.show(block=block_plots)
    
    gaintable = create_gaintable_from_blockvisibility(block_vis)
    mpccal_skymodel = initialize_skymodel_voronoi(model, ical_components, ical_skymodel[0].gaintable)
    
    model = create_empty_image_like(mpccal_skymodel[0].image)
    
    future_model = arlexecute.scatter(model)
    future_theta_list = arlexecute.scatter(mpccal_skymodel)
    future_vis = arlexecute.scatter(all_skymodel_vis)
    result = mpccal_skymodel_list_arlexecute_workflow(future_vis, future_model, future_theta_list,
                                                      mpccal_progress=partial(progress, context='MPCCAL'),
                                                      nmajor=args.mpccal_nmajor,
                                                      context='2d',
                                                      algorithm='hogbom',
                                                      fractional_threshold=0.3, threshold=0.01,
                                                      gain=0.1, niter=1000,
                                                      psf_support=512,
                                                      deconvolve_facets=8,
                                                      deconvolve_overlap=16,
                                                      deconvolve_taper='tukey')
    
    (mpccal_skymodel, mpccal_residual) = arlexecute.compute(result, sync=True)
    print(qa_image(residual, context='MPCCal residual image'))
    
    print('mpccal finished')
    
    mpccal_combined_model = calculate_skymodel_equivalent_image(mpccal_skymodel)
    print(qa_image(mpccal_combined_model, context='MPCCAL combined model'))
    psf_obs = invert_list_arlexecute_workflow([future_vis], [future_model], context='2d', dopsf=True)
    result = restore_list_arlexecute_workflow([mpccal_combined_model], psf_obs, [(mpccal_residual, 0.0)])
    result = arlexecute.compute(result, sync=True)
    mpccal_restored = result[0]
    
    mpccal_finding_threshold = args.mpccal_threshold * qa_image(mpccal_restored).data['medianabs']
    mpccal_components = find_skycomponents(mpccal_restored, fwhm=2,
                                         threshold=mpccal_finding_threshold, npixels=12)
    mpccal_components = sorted(mpccal_components, key=lambda comp: numpy.max(comp.flux), reverse=True)
    print("Number of components in MPCCAL %d" % len(mpccal_components))

    print(qa_image(mpccal_restored, context='MPCCAL restored image'))
    show_image(mpccal_restored, title='MPCCAL restored image', components=mpccal_components)
    plt.show(block=block_plots)
    export_image_to_fits(mpccal_restored, arl_path('test_results/low-sims-mpc-mpccal-restored_%.1frmax.fits' % rmax))
    
    difference_image = copy_image(mpccal_restored)
    difference_image.data -= ical_restored.data

    print(qa_image(difference_image, context='MPCCAL - ICAL image'))
    show_image(difference_image, title='MPCCAL - ICAL image', components=ical_components)
    plt.show(block=block_plots)
    export_image_to_fits(difference_image, arl_path('test_results/low-sims-mpc-mpccal-ical-restored_%.1frmax.fits' %
                                                    rmax))

    from processing_components.simulation.ionospheric_screen import grid_gaintable_to_screen, plot_gaintable_on_screen
    from processing_components.image.operations import create_empty_image_like
    
    newscreen = create_empty_image_like(screen)
    gaintables = [sm.gaintable for sm in mpccal_skymodel]
    newscreen, weights = grid_gaintable_to_screen(block_vis, gaintables, newscreen)
    plot_gaintable_on_screen(block_vis, gaintables)
    plt.savefig(arl_path('test_results/low-sims-mpc-mpccal-screen_%.1frmax.png' % rmax))
    plt.show(block=block_plots)
    export_image_to_fits(newscreen, arl_path('test_results/low-sims-mpc-mpccal-screen_%.1frmax.fits' % rmax))
    export_image_to_fits(weights, arl_path('test_results/low-sims-mpc-mpccal-screenweights_%.1frmax.fits' % rmax))
    print(qa_image(weights))
    print(qa_image(newscreen))
    
    arlexecute.close()
