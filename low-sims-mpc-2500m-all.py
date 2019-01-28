# coding: utf-8

# ## Simulate non-isoplanatic imaging for LOW at 100MHz.
# 
# ### A set of model components are drawn from GLEAM. An ionospheric screen model is used to calculate the pierce points of the two stations in an interferometer for a given component. The model visibilities are calculated directly, and screen phase applied to obtain the corrupted visibility.

# In[ ]:

import logging
import sys

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.memory_data_models import SkyModel
from data_models.parameters import arl_path
from data_models.polarisation import PolarisationFrame

from processing_library.image.operations import create_empty_image_like

from workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow
from workflows.arlexecute.pipelines.pipeline_mpccal_arlexecute import mpccal_skymodel_list_arlexecute_workflow
from workflows.serial.imaging.imaging_serial import weight_list_serial_workflow, taper_list_serial_workflow
from workflows.arlexecute.skymodel.skymodel_arlexecute import predict_skymodel_list_arlexecute_workflow
from workflows.arlexecute.imaging.imaging_arlexecute import restore_list_arlexecute_workflow

from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.execution_support.dask_init import get_dask_Client
from wrappers.arlexecute.skymodel.operations import calculate_skymodel_equivalent_image
from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility
from wrappers.serial.image.operations import import_image_from_fits
from wrappers.serial.image.operations import qa_image
from wrappers.serial.imaging.base import create_image_from_visibility, advise_wide_field
from wrappers.serial.imaging.primary_beams import create_low_test_beam
from wrappers.serial.simulation.ionospheric_screen import create_gaintable_from_screen
from wrappers.serial.simulation.testing_support import create_named_configuration, \
    create_low_test_skycomponents_from_gleam
from wrappers.serial.skycomponent.operations import apply_beam_to_skycomponent
from wrappers.serial.skycomponent.operations import filter_skycomponents_by_flux
from wrappers.serial.visibility.base import create_blockvisibility, copy_visibility
from wrappers.serial.visibility.operations import qa_visibility
from wrappers.arlexecute.skymodel.operations import initialize_skymodel_voronoi
from wrappers.arlexecute.skycomponent.operations import remove_neighbouring_components, \
    voronoi_decomposition, find_skycomponents
from wrappers.arlexecute.calibration.operations import create_gaintable_from_blockvisibility

if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stdout))
    
    use_dask = True
    n_workers = 8

    if use_dask:
        c = get_dask_Client(memory_limit=64 * 1024 * 1024 * 1024, n_workers=n_workers, threads_per_worker=1)
        arlexecute.set_client(c)
    else:
        arlexecute.set_client(use_dask=False
                              )
    # Initialise logging on the workers. This appears to only work using the process scheduler.
    
    # Set up the observation: 10 minutes at transit, with 10s integration.
    # Skip 5/6 points to avoid out station redundancy
    
    nfreqwin = 1
    ntimes = 3
    rmax = 2500.0
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
    
    blockvis = create_blockvisibility(
        low,
        times,
        frequency=frequency,
        channel_bandwidth=channel_bandwidth,
        weight=1.0,
        phasecentre=phasecentre,
        polarisation_frame=PolarisationFrame("stokesI"),
        zerow=True)
    
    vis = convert_blockvisibility_to_visibility(blockvis)
    advice = advise_wide_field(vis, guard_band_image=2.0, delA=0.02)
    
    cellsize = advice['cellsize']
    vis_slices = advice['vis_slices']
    npixel = advice['npixels2']
    
    small_model = create_image_from_visibility(
        blockvis,
        npixel=512,
        frequency=frequency,
        nchan=nfreqwin,
        cellsize=cellsize,
        phasecentre=phasecentre)
    
    vis.data['imaging_weight'][...] = vis.data['weight'][...]
    vis = weight_list_serial_workflow([vis], [small_model])[0]
    vis = taper_list_serial_workflow([vis], 3 * cellsize)[0]
    
    block_vis = convert_visibility_to_blockvisibility(vis)
    
    # ### Generate the model from the GLEAM catalog, including application of the primary beam.
    
    flux_limit = 0.2
    beam = create_image_from_visibility(
        blockvis,
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
        radius=0.2)
    
    all_components = apply_beam_to_skycomponent(original_gleam_components, beam)
    all_components = filter_skycomponents_by_flux(all_components, flux_min=flux_limit)
    voronoi_components = filter_skycomponents_by_flux(all_components, flux_min=0.7)
    
    print("Number of components %d" % len(all_components))
    screen = import_image_from_fits(arl_path('data/models/test_mpc_screen.fits'))
    all_gaintables = create_gaintable_from_screen(blockvis, all_components, screen)
    
    gleam_skymodel_noniso = [SkyModel(components=[all_components[i]], gaintable=all_gaintables[i])
                             for i, sm in enumerate(all_components)]
    
    # ### Now predict the visibility for each skymodel and apply the gaintable for that skymodel,
    # returning a list of visibilities, one for each skymodel. We then sum these to obtain
    # the total predicted visibility. All images and skycomponents in the same skymodel
    # get the same gaintable applied which means that in this case each skycomponent has a separate gaintable.
    
    all_skymodel_noniso_blockvis = copy_visibility(blockvis, zero=True)
    all_skymodel_noniso_vis = convert_blockvisibility_to_visibility(all_skymodel_noniso_blockvis)
    
    ngroup = n_workers
    future_vis = arlexecute.scatter(all_skymodel_noniso_vis)
    chunks = [gleam_skymodel_noniso[i:i + ngroup] for i in range(0, len(gleam_skymodel_noniso), ngroup)]
    for chunk in chunks:
        result = predict_skymodel_list_arlexecute_workflow(future_vis, chunk, context='2d', docal=True)
        work_vis = arlexecute.compute(result, sync=True)
        for w in work_vis:
            all_skymodel_noniso_vis.data['vis'] += w.data['vis']
        assert numpy.max(numpy.abs(all_skymodel_noniso_vis.data['vis'])) > 0.0

    all_skymodel_noniso_blockvis = convert_visibility_to_blockvisibility(all_skymodel_noniso_vis)
    
    # ### Remove weaker of components that are too close (0.02 rad)
    idx, voronoi_components = remove_neighbouring_components(voronoi_components, 0.02)
    print(len(voronoi_components))
    nsources = len(voronoi_components)

    model = create_image_from_visibility(
        blockvis,
        npixel=npixel,
        frequency=frequency,
        nchan=nfreqwin,
        cellsize=cellsize,
        phasecentre=phasecentre)
    
    vor, vor_array = voronoi_decomposition(model, voronoi_components)
    vor_image = create_empty_image_like(model)
    vor_image.data[...] = vor_array
    
    gaintable = create_gaintable_from_blockvisibility(blockvis)
    theta_list = initialize_skymodel_voronoi(model, voronoi_components, gaintable)
    
    model = create_empty_image_like(theta_list[0].image)
    
    # End of setup, start of processing

    def progress(res, tl_list, gt_list, it):
        import matplotlib.pyplot as plt
        print('Iteration %d' % it)
    
        print('Length of theta = %d' % len(tl_list))
    
        print(qa_image(res, context='Residual image: iteration %d' % it))
        export_image_to_fits(res, "results/low-sims-mpc-residual_iteration%d_rmax%.1f.hdf5" %
                             (it, rmax))
    
        combined_model = calculate_skymodel_equivalent_image(tl_list)
        print(qa_image(combined_model, context='Combined model: iteration %d' % it))
        export_image_to_fits(combined_model, "results/low-sims-mpc-model_iteration%d_rmax%.1f.hdf5" %
                             (it, rmax))
    
        plt.clf()
        for i in range(len(tl_list)):
            plt.plot(numpy.angle(tl_list[i].gaintable.gain[:, :, 0, 0, 0]).flatten(),
                     numpy.angle(gt_list[i]['T'].gain[:, :, 0, 0, 0]).flatten(),
                     '.')
        plt.xlabel('Current phase')
        plt.ylabel('Update to phase')
        plt.xlim([-numpy.pi, numpy.pi])
        plt.ylim([-numpy.pi, numpy.pi])
        plt.title("MPCCal %dsources iteration%d: Change in phase" % (nsources, it))
        plt.savefig("figures/low-sims-mpc-skymodel-phase-change_iteration%d.jpg" % (it))
        plt.show()
        return tl_list


    future_model = arlexecute.scatter(model)
    future_theta_list = arlexecute.scatter(theta_list)
    future_vis = arlexecute.scatter(all_skymodel_noniso_vis)
    result = mpccal_skymodel_list_arlexecute_workflow(future_vis, future_model, future_theta_list,
                                                      mpccal_progress=progress,
                                                      nmajor=5,
                                                      context='2d',
                                                      algorithm='hogbom',
                                                      fractional_threshold=0.3, threshold=0.1,
                                                      gain=0.1, niter=1000,
                                                      psf_support=512,
                                                      deconvolve_facets=8,
                                                      deconvolve_overlap=16,
                                                      deconvolve_taper='tukey')
    
    (theta_list, residual) = arlexecute.compute(result, sync=True)
    print(qa_image(residual, context='MPCCal residual image'))
    
    print(theta_list[0].gaintable.gain)

    print('mpccal finished')

    combined_model = calculate_skymodel_equivalent_image(theta_list)
    print(qa_image(combined_model, context='MPCCal combined model'))
    psf_obs = invert_list_arlexecute_workflow([future_vis], [future_model], context='2d', dopsf=True)
    result = restore_list_arlexecute_workflow([combined_model], psf_obs, [(residual, 0.0)])
    result = arlexecute.compute(result, sync=True)
    
    print(qa_image(result[0], context='MPCCal restored image'))
    from wrappers.arlexecute.image.operations import export_image_to_fits
    export_image_to_fits(result[0], 'mpccal.fits')
    
    recovered_mpccal_components = find_skycomponents(result[0], fwhm=2, threshold=0.15, npixels=12)
    print(len(recovered_mpccal_components))
    print(recovered_mpccal_components[0])
    
    from processing_components.simulation.ionospheric_screen import grid_gaintable_to_screen
    from processing_components.image.operations import create_empty_image_like
    
    newscreen = create_empty_image_like(screen)
    gaintables = [th.gaintable for th in theta_list]
    newscreen, weights = grid_gaintable_to_screen(blockvis, gaintables, newscreen)
    export_image_to_fits(newscreen, 'mpccal_screen.fits')
    export_image_to_fits(weights, 'mpccal_screenweights.fits')
    print(qa_image(weights))
    print(qa_image(newscreen))

    arlexecute.close()
