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
from data_models.parameters import arl_path
from data_models.polarisation import PolarisationFrame
from processing_library.image.operations import create_empty_image_like, copy_image
from workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow
from workflows.arlexecute.imaging.imaging_arlexecute import restore_list_arlexecute_workflow
from workflows.arlexecute.pipelines.pipeline_mpccal_arlexecute import mpccal_skymodel_list_arlexecute_workflow
from workflows.arlexecute.skymodel.skymodel_arlexecute import predict_skymodel_list_arlexecute_workflow
from workflows.serial.imaging.imaging_serial import weight_list_serial_workflow, taper_list_serial_workflow
from wrappers.arlexecute.calibration.operations import create_gaintable_from_blockvisibility
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.execution_support.dask_init import get_dask_Client
from wrappers.arlexecute.image.operations import export_image_to_fits
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
from wrappers.serial.simulation.ionospheric_screen import grid_gaintable_to_screen, plot_gaintable_on_screen
from wrappers.serial.simulation.testing_support import create_low_test_skycomponents_from_gleam
from processing_components.simulation.configurations import create_named_configuration
from wrappers.serial.skycomponent.operations import apply_beam_to_skycomponent, insert_skycomponent
from wrappers.serial.skycomponent.operations import filter_skycomponents_by_flux
from wrappers.serial.visibility.base import create_blockvisibility, copy_visibility

if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stdout))
    
    import argparse
    
    # Default parameters produce a good MPCCAL image with rmax=2500.0
    # For rmax=5000.0, use
    parser = argparse.ArgumentParser(description='MPCCAL pipeline example')
    
    parser.add_argument('--use_dask', type=str, default='True', help='Use Dask?')
    parser.add_argument('--nworkers', type=int, default=8, help='Number of workers')
    parser.add_argument('--memory', type=int, default=32, help='Memory per worker')

    parser.add_argument('--ical_nmajor', type=int, default=10, help='Number of major cycles for ICAL')
    parser.add_argument('--mpccal_nmajor', type=int, default=20, help='Number of major cycles for MPCCAL')
    parser.add_argument('--ntimes', type=int, default=3, help='Number of hour angles')
    parser.add_argument('--block', type=str, default='False', help='Block plotting output until keypress?')
    parser.add_argument('--context', type=str, default='2d', help='Imaging context: 2d|timeslice|wstack')
    parser.add_argument('--rmax', type=float, default=2500.0, help='Maximum baseline (m)')
    parser.add_argument('--flux_limit', type=float, default=0.2, help='Flux limit for GLEAM components')
    parser.add_argument('--finding_threshold', type=float, default=1.0, help='Source finding threshold (Jy)')
    parser.add_argument('--ninitial', type=int, default=1, help='Number of initial components to use')
    args = parser.parse_args()
    block_plots = args.block == 'True'
    
    if args.use_dask:
        c = get_dask_Client(n_workers=args.nworkers, memory_limit=args.memory * 1024 * 1024 * 1024)
        arlexecute.set_client(c)
    else:
        arlexecute.set_client(use_dask=False)
    #######################################################################################################
    # Set up the observation: 10 minutes at transit, with 10s integration.
    # Skip 5/6 points to avoid outstation redundancy. Apply uniform weighting.
    
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
    ### Generate the component model from the GLEAM catalog, including application of the primary beam. Read the
    # phase screen and calculate the gaintable for each component.
    flux_limit = args.flux_limit
    beam = create_image_from_visibility(
        block_vis,
        npixel=npixel,
        frequency=frequency,
        nchan=nfreqwin,
        cellsize=cellsize,
        phasecentre=phasecentre)
    beam = create_low_test_beam(beam, use_local=False
    
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
    
    #######################################################################################################
    # Calculate visibility by using the predict_skymodel function which applies a different gaintable table
    # for each skymodel. We do the calculation in chunks of nworkers skymodels.
    all_skymodel_blockvis = copy_visibility(block_vis, zero=True)
    all_skymodel_vis = convert_blockvisibility_to_visibility(all_skymodel_blockvis)
    
    ngroup = 8
    future_vis = arlexecute.scatter(all_skymodel_vis)
    chunks = [all_skymodel[i:i + ngroup] for i in range(0, len(all_skymodel), ngroup)]
    for chunk in chunks:
        result = predict_skymodel_list_arlexecute_workflow(future_vis, chunk, context='2d', docal=True)
        work_vis = arlexecute.compute(result, sync=True)
        for w in work_vis:
            all_skymodel_vis.data['vis'] += w.data['vis']
        assert numpy.max(numpy.abs(all_skymodel_vis.data['vis'])) > 0.0
    
    all_skymodel_blockvis = convert_visibility_to_blockvisibility(all_skymodel_vis)
    
    #######################################################################################################
    # Now proceed to run MPCCAL in ICAL mode i.e. with only one skymodel
    def progress(res, tl_list, gt_list, it, context='MPCCAL'):
        print('Iteration %d' % it)
        
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
        
    all_fluxes = [sm.components[0].flux[0, 0] for sm in all_skymodel]
    
    null_gaintable = create_gaintable_from_blockvisibility(block_vis)
    
    #######################################################################################################
    # Set up and run MPCCAL in ICAL mode i.e. there is only one skymodel
    model = create_image_from_visibility(block_vis, npixel=npixel, frequency=frequency, nchan=nfreqwin,
                                         cellsize=cellsize, phasecentre=phasecentre)

    ical_skymodel = [SkyModel(components=all_components[:args.ninitial], gaintable=null_gaintable,
                              image=model)]

    future_vis = arlexecute.scatter(all_skymodel_vis)
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
    
    print('ical finished')
    
    combined_model = calculate_skymodel_equivalent_image(ical_skymodel)
    print(qa_image(combined_model, context='ICAL combined model'))
    psf_obs = invert_list_arlexecute_workflow([future_vis], [future_model], context='2d', dopsf=True)
    result = restore_list_arlexecute_workflow([combined_model], psf_obs, [(residual, 0.0)])
    result = arlexecute.compute(result, sync=True)
    ical_restored = result[0]
    
    export_image_to_fits(ical_restored, arl_path('test_results/low-sims-mpc-ical-restored_%.1frmax.fits' % rmax))
    
    #######################################################################################################
    # Now set up the skymodels for MPCCAL. We find the brightest components in the ICAL image, remove
    # sources that are too close to another stronger source, and then use these to set up
    # a Voronoi tesselation to define the skymodel masks
    
    ical_components = find_skycomponents(ical_restored, fwhm=2,
                                         threshold=args.finding_threshold, npixels=12)
    for comp in all_components[:args.ninitial]:
        ical_components.append(comp)
        
    # ### Remove weaker of components that are too close (0.02 rad)
    idx, ical_components = remove_neighbouring_components(ical_components, 0.02)
    ical_components = sorted(ical_components, key=lambda comp: numpy.max(comp.flux), reverse=True)
    print("Voronoi decomposition based on %d point sources" % len(ical_components))
    
    print(qa_image(ical_restored, context='ICAL restored image'))
    show_image(ical_restored, title='ICAL restored image', vmax=0.3, vmin=-0.03)
    show_image(ical_restored, title='ICAL restored image', components=ical_components, vmax=0.3, vmin=-0.03)
    plt.show(block=block_plots)

    gaintable = create_gaintable_from_blockvisibility(block_vis)
    mpccal_skymodel = initialize_skymodel_voronoi(model, ical_components,
                                                  ical_skymodel[0].gaintable)
    
    ical_fluxes = [comp.flux[0, 0] for comp in ical_components]
    plt.clf()
    plt.semilogy(numpy.arange(len(all_fluxes)), all_fluxes, marker='.')
    plt.semilogy(numpy.arange(len(ical_fluxes)), ical_fluxes, marker='.')
    plt.title('All component fluxes')
    plt.ylabel('Flux (Jy)')
    plt.show(block=block_plots)
    
    #######################################################################################################
    # Now we can run MPCCAL with O(10) distinct masks
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
    print(qa_image(mpccal_residual, context='MPCCal residual image'))
    
    print('mpccal finished')

    mpccal_combined_model = calculate_skymodel_equivalent_image(mpccal_skymodel)
    mpccal_combined_model = insert_skycomponent(mpccal_combined_model, ical_components)
    print(qa_image(mpccal_combined_model, context='MPCCAL combined model'))
    
    psf_obs = invert_list_arlexecute_workflow([future_vis], [future_model], context='2d', dopsf=True)
    result = restore_list_arlexecute_workflow([mpccal_combined_model], psf_obs, [(mpccal_residual, 0.0)])
    result = arlexecute.compute(result, sync=True)
    mpccal_restored = result[0]
    
    mpccal_components = find_skycomponents(mpccal_restored, fwhm=2,
                                           threshold=args.finding_threshold, npixels=12)
    mpccal_components = sorted(mpccal_components, key=lambda comp: numpy.max(comp.flux), reverse=True)
    print("Number of components in MPCCAL %d" % len(mpccal_components))
    
    print(qa_image(mpccal_restored, context='MPCCAL restored image'))
    show_image(mpccal_restored, title='MPCCAL restored image', vmax=0.3, vmin=-0.03)
    show_image(mpccal_restored, title='MPCCAL restored image', components=mpccal_components, vmax=0.3, vmin=-0.03)
    plt.show(block=block_plots)
    export_image_to_fits(mpccal_restored, arl_path('test_results/low-sims-mpc-mpccal-restored_%.1frmax.fits' % rmax))
    
    mpccal_fluxes = [comp.flux[0, 0] for comp in mpccal_components]
    plt.clf()
    plt.semilogy(numpy.arange(len(all_fluxes)), all_fluxes, marker='.', label='Original')
    plt.semilogy(numpy.arange(len(ical_fluxes)), ical_fluxes, marker='.', label='ICAL')
    plt.semilogy(numpy.arange(len(mpccal_fluxes)), mpccal_fluxes, marker='.', label='MPCCAL')
    plt.title('All component fluxes')
    plt.ylabel('Flux (Jy)')
    plt.legend()
    plt.show(block=block_plots)
    
    difference_image = copy_image(mpccal_restored)
    difference_image.data -= ical_restored.data
    
    print(qa_image(difference_image, context='MPCCAL - ICAL image'))
    show_image(difference_image, title='MPCCAL - ICAL image', components=ical_components)
    plt.show(block=block_plots)
    export_image_to_fits(difference_image, arl_path('test_results/low-sims-mpc-mpccal-ical-restored_%.1frmax.fits' %
                                                    rmax))
    
    newscreen = create_empty_image_like(screen)
    gaintables = [sm.gaintable for sm in mpccal_skymodel]
    newscreen, weights = grid_gaintable_to_screen(block_vis, gaintables, newscreen)
    export_image_to_fits(newscreen, arl_path('test_results/low-sims-mpc-mpccal-screen_%.1frmax.fits' % rmax))
    export_image_to_fits(weights, arl_path('test_results/low-sims-mpc-mpccal-screenweights_%.1frmax.fits' % rmax))
    print(qa_image(weights))
    print(qa_image(newscreen))
    
    plot_gaintable_on_screen(block_vis, gaintables)
    plt.savefig(arl_path('test_results/low-sims-mpc-mpccal-screen_%.1frmax.png' % rmax))
    plt.show(block=block_plots)
    
    arlexecute.close()
