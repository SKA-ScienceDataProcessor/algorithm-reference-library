import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.memory_data_models import SkyModel
from data_models.parameters import arl_path
from data_models.polarisation import PolarisationFrame
from processing_components.image.operations import create_empty_image_like
from processing_components.simulation.configurations import create_named_configuration
from processing_components.simulation.ionospheric_screen import grid_gaintable_to_screen
from workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow
from workflows.arlexecute.imaging.imaging_arlexecute import restore_list_arlexecute_workflow
from workflows.arlexecute.pipelines.pipeline_mpccal_arlexecute import mpccal_skymodel_list_arlexecute_workflow
from workflows.arlexecute.skymodel.skymodel_arlexecute import predict_skymodel_list_arlexecute_workflow
from workflows.serial.imaging.imaging_serial import weight_list_serial_workflow, taper_list_serial_workflow
from wrappers.arlexecute.execution_support.arlexecutebase import ARLExecuteBase
from wrappers.arlexecute.execution_support.dask_init import get_dask_Client
from wrappers.arlexecute.image.operations import export_image_to_fits
from wrappers.arlexecute.skycomponent.operations import remove_neighbouring_components, \
    find_skycomponents
from wrappers.arlexecute.skymodel.operations import calculate_skymodel_equivalent_image
from wrappers.arlexecute.skymodel.operations import initialize_skymodel_voronoi
from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility
from wrappers.serial.image.operations import import_image_from_fits
from wrappers.serial.imaging.base import create_image_from_visibility, advise_wide_field
from wrappers.serial.imaging.primary_beams import create_low_test_beam
from wrappers.serial.simulation.ionospheric_screen import create_gaintable_from_screen
from wrappers.serial.simulation.testing_support import create_low_test_skycomponents_from_gleam
from wrappers.serial.skycomponent.operations import apply_beam_to_skycomponent
from wrappers.serial.skycomponent.operations import filter_skycomponents_by_flux
from wrappers.serial.visibility.base import create_blockvisibility

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestPipelineMPC(unittest.TestCase):
    def setUp(self):
        
        client = get_dask_Client(memory_limit=4 * 1024 * 1024 * 1024, n_workers=4, dashboard_address=None)
        global arlexecute
        arlexecute = ARLExecuteBase(use_dask=True)
        arlexecute.set_client(client)
        
        self.persist = False

    def tearDown(self):
        global arlexecute
        arlexecute.close()
        del arlexecute

    def progress(self, res, tl_list, gt_list, it):
        """Write progress information
        
        Cannot use this if using Dask
        
        :param res: Residual image
        :param tl_list: Theta list
        :param gt_list: Gaintable list
        :param it: iteration
        :return:
        """
        
        import matplotlib.pyplot as plt
        plt.clf()
        for i in range(len(tl_list)):
            plt.plot(numpy.angle(tl_list[i].gaintable.gain[:, :, 0, 0, 0]).flatten(),
                     numpy.angle(gt_list[i]['T'].gain[:, :, 0, 0, 0]).flatten(),
                     '.')
        plt.xlabel('Current phase')
        plt.ylabel('Update to phase')
        # plt.xlim([-numpy.pi, numpy.pi])
        # plt.ylim([-numpy.pi, numpy.pi])
        plt.title("MPCCal iteration%d: Change in phase" % (it))
        plt.show()
        
        return tl_list
    
    def actualSetup(self, nsources=None, nvoronoi=None):
        
        n_workers = 8
        
        # Set up the observation: 10 minutes at transit, with 10s integration.
        # Skip 5/6 points to avoid outstation redundancy
        
        nfreqwin = 1
        ntimes = 3
        self.rmax = 2500.0
        dec = -40.0 * u.deg
        frequency = [1e8]
        channel_bandwidth = [0.1e8]
        times = numpy.linspace(-10.0, 10.0, ntimes) * numpy.pi / (3600.0 * 12.0)
        
        phasecentre = SkyCoord(ra=+0.0 * u.deg, dec=dec, frame='icrs', equinox='J2000')
        low = create_named_configuration('LOWBD2', rmax=self.rmax)
        
        centre = numpy.mean(low.xyz, axis=0)
        distance = numpy.hypot(low.xyz[:, 0] - centre[0],
                               low.xyz[:, 1] - centre[1],
                               low.xyz[:, 2] - centre[2])
        lowouter = low.data[distance > 1000.0][::6]
        lowcore = low.data[distance < 1000.0][::3]
        low.data = numpy.hstack((lowcore, lowouter))
        
        blockvis = create_blockvisibility(low, times, frequency=frequency, channel_bandwidth=channel_bandwidth,
                                          weight=1.0, phasecentre=phasecentre,
                                          polarisation_frame=PolarisationFrame("stokesI"), zerow=True)
        
        vis = convert_blockvisibility_to_visibility(blockvis)
        advice = advise_wide_field(vis, guard_band_image=2.0, delA=0.02)
        
        cellsize = advice['cellsize']
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
        
        blockvis = convert_visibility_to_blockvisibility(vis)
        
        # ### Generate the model from the GLEAM catalog, including application of the primary beam.
        
        beam = create_image_from_visibility(blockvis, npixel=npixel, frequency=frequency,
                                            nchan=nfreqwin, cellsize=cellsize, phasecentre=phasecentre)
        beam = create_low_test_beam(beam, use_local=False)
        
        flux_limit = 0.5
        original_gleam_components = create_low_test_skycomponents_from_gleam(flux_limit=flux_limit,
                                                                             phasecentre=phasecentre,
                                                                             frequency=frequency,
                                                                             polarisation_frame=PolarisationFrame(
                                                                                 'stokesI'),
                                                                             radius=0.15)
        
        all_components = apply_beam_to_skycomponent(original_gleam_components, beam)
        all_components = filter_skycomponents_by_flux(all_components, flux_min=flux_limit)
        voronoi_components = filter_skycomponents_by_flux(all_components, flux_min=1.5)
        
        def max_flux(elem):
            return numpy.max(elem.flux)
        
        voronoi_components = sorted(voronoi_components, key=max_flux, reverse=True)
        
        if nsources is not None:
            all_components = [all_components[0]]
        
        if nvoronoi is not None:
            voronoi_components = [voronoi_components[0]]
        
        self.screen = import_image_from_fits(arl_path('data/models/test_mpc_screen.fits'))
        all_gaintables = create_gaintable_from_screen(blockvis, all_components, self.screen)
        
        gleam_skymodel_noniso = [SkyModel(components=[all_components[i]], gaintable=all_gaintables[i])
                                 for i, sm in enumerate(all_components)]
        
        # ### Now predict the visibility for each skymodel and apply the gaintable for that skymodel,
        # returning a list of visibilities, one for each skymodel. We then sum these to obtain
        # the total predicted visibility. All images and skycomponents in the same skymodel
        # get the same gaintable applied which means that in this case each skycomponent has a separate gaintable.
        
        self.all_skymodel_noniso_vis = convert_blockvisibility_to_visibility(blockvis)
        
        ngroup = n_workers
        future_vis = arlexecute.scatter(self.all_skymodel_noniso_vis)
        chunks = [gleam_skymodel_noniso[i:i + ngroup] for i in range(0, len(gleam_skymodel_noniso), ngroup)]
        for chunk in chunks:
            result = predict_skymodel_list_arlexecute_workflow(future_vis, chunk, context='2d', docal=True)
            work_vis = arlexecute.compute(result, sync=True)
            for w in work_vis:
                self.all_skymodel_noniso_vis.data['vis'] += w.data['vis']
            assert numpy.max(numpy.abs(self.all_skymodel_noniso_vis.data['vis'])) > 0.0
        
        self.all_skymodel_noniso_blockvis = convert_visibility_to_blockvisibility(self.all_skymodel_noniso_vis)
        
        # ### Remove weaker of components that are too close (0.02 rad)
        idx, voronoi_components = remove_neighbouring_components(voronoi_components, 0.02)
        
        model = create_image_from_visibility(blockvis, npixel=npixel, frequency=frequency,
                                             nchan=nfreqwin, cellsize=cellsize, phasecentre=phasecentre)
        
        # Use the gaintable for the brightest component as the starting gaintable
        all_gaintables[0].gain[...] = numpy.conjugate(all_gaintables[0].gain[...])
        all_gaintables[0].gain[...] = 1.0 + 0.0j
        self.theta_list = initialize_skymodel_voronoi(model, voronoi_components, all_gaintables[0])
    
    # End of setup, start of processing]
    
    def test_time_setup(self):
        self.actualSetup(nsources=1, nvoronoi=1)
        pass
    
    def test_mpccal_ICAL_manysources(self):
        
        self.actualSetup(nvoronoi=1)
        
        model = create_empty_image_like(self.theta_list[0].image)
        
        if arlexecute.using_dask:
            progress = None
        else:
            progress = self.progress
        
        future_vis = arlexecute.scatter(self.all_skymodel_noniso_vis)
        future_model = arlexecute.scatter(model)
        future_theta_list = arlexecute.scatter(self.theta_list)
        result = mpccal_skymodel_list_arlexecute_workflow(future_vis, future_model, future_theta_list,
                                                          mpccal_progress=progress,
                                                          nmajor=10,
                                                          context='2d',
                                                          algorithm='hogbom',
                                                          scales=[0, 3, 10],
                                                          fractional_threshold=0.3, threshold=0.05,
                                                          gain=0.1, niter=1000, psf_support=512,
                                                          deconvolve_facets=8, deconvolve_overlap=16,
                                                          deconvolve_taper='tukey')
        
        (self.theta_list, residual) = arlexecute.compute(result, sync=True)
        
        combined_model = calculate_skymodel_equivalent_image(self.theta_list)
        
        psf_obs = invert_list_arlexecute_workflow([self.all_skymodel_noniso_vis], [model], context='2d', dopsf=True)
        result = restore_list_arlexecute_workflow([combined_model], psf_obs, [(residual, 0.0)])
        result = arlexecute.compute(result, sync=True)
        
        if self.persist: export_image_to_fits(residual, arl_path('test_results/test_mpccal_ical_many_residual.fits'))
        if self.persist: export_image_to_fits(result[0], arl_path('test_results/test_mpccal_ical_many_restored.fits'))
        if self.persist: export_image_to_fits(combined_model, arl_path('test_results/test_mpccal_ical_many_deconvolved.fits'))
        
        recovered_mpccal_components = find_skycomponents(result[0], fwhm=2, threshold=0.32, npixels=12)
        
        def max_flux(elem):
            return numpy.max(elem.flux)
        
        recovered_mpccal_components = sorted(recovered_mpccal_components, key=max_flux, reverse=True)
        
        assert recovered_mpccal_components[0].name == 'Segment 5', recovered_mpccal_components[0].name
        assert numpy.abs(recovered_mpccal_components[0].flux[0, 0] - 7.318399477857547) < 1e-7, \
            recovered_mpccal_components[0].flux[0, 0]
        
        newscreen = create_empty_image_like(self.screen)
        gaintables = [th.gaintable for th in self.theta_list]
        newscreen, weights = grid_gaintable_to_screen(self.all_skymodel_noniso_blockvis, gaintables, newscreen)
        if self.persist: export_image_to_fits(newscreen, arl_path('test_results/test_mpccal_ical_many_screen.fits'))
        if self.persist: export_image_to_fits(weights, arl_path('test_results/test_mpccal_ical_many_screenweights.fits'))
        
        arlexecute.close()
    
    def test_mpccal_ICAL_onesource(self):
        
        self.actualSetup(nsources=1, nvoronoi=1)
        
        model = create_empty_image_like(self.theta_list[0].image)
        
        if arlexecute.using_dask:
            progress = None
        else:
            progress = self.progress
        
        future_vis = arlexecute.scatter(self.all_skymodel_noniso_vis)
        future_model = arlexecute.scatter(model)
        future_theta_list = arlexecute.scatter(self.theta_list)
        result = mpccal_skymodel_list_arlexecute_workflow(future_vis, future_model, future_theta_list,
                                                          mpccal_progress=progress,
                                                          nmajor=10,
                                                          context='2d',
                                                          algorithm='hogbom',
                                                          scales=[0, 3, 10],
                                                          fractional_threshold=0.3, threshold=0.01,
                                                          gain=0.1, niter=1000, psf_support=256)
        
        (self.theta_list, residual) = arlexecute.compute(result, sync=True)
        
        combined_model = calculate_skymodel_equivalent_image(self.theta_list)
        
        psf_obs = invert_list_arlexecute_workflow([self.all_skymodel_noniso_vis], [model], context='2d', dopsf=True)
        result = restore_list_arlexecute_workflow([combined_model], psf_obs, [(residual, 0.0)])
        result = arlexecute.compute(result, sync=True)
        
        if self.persist: export_image_to_fits(residual, arl_path('test_results/test_mpccal_ical_onesource_residual.fits'))
        if self.persist: export_image_to_fits(result[0], arl_path('test_results/test_mpccal_ical_onesource_restored.fits'))
        if self.persist: export_image_to_fits(combined_model, arl_path('test_results/test_mpccal_ical_onesource_deconvolved.fits'))
        
        recovered_mpccal_components = find_skycomponents(result[0], fwhm=2, threshold=0.32, npixels=12)
        
        def max_flux(elem):
            return numpy.max(elem.flux)
        
        recovered_mpccal_components = sorted(recovered_mpccal_components, key=max_flux, reverse=True)
        
        assert recovered_mpccal_components[0].name == 'Segment 0', recovered_mpccal_components[0].name
        assert numpy.abs(recovered_mpccal_components[0].flux[0, 0] - 1.2373939075803901) < 1e-6, \
            recovered_mpccal_components[0].flux[0, 0]
        
        newscreen = create_empty_image_like(self.screen)
        gaintables = [th.gaintable for th in self.theta_list]
        newscreen, weights = grid_gaintable_to_screen(self.all_skymodel_noniso_blockvis, gaintables, newscreen)
        if self.persist: export_image_to_fits(newscreen, arl_path('test_results/test_mpccal_ical_onesource_screen.fits'))
        if self.persist: export_image_to_fits(weights, arl_path('test_results/test_mpccal_ical_onesource_screenweights.fits'))
        
        arlexecute.close()

    def test_mpccal_MPCCAL_manysources(self):
    
        self.actualSetup()
    
        model = create_empty_image_like(self.theta_list[0].image)
    
        if arlexecute.using_dask:
            progress = None
        else:
            progress = self.progress
    
        future_vis = arlexecute.scatter(self.all_skymodel_noniso_vis)
        future_model = arlexecute.scatter(model)
        future_theta_list = arlexecute.scatter(self.theta_list)
        result = mpccal_skymodel_list_arlexecute_workflow(future_vis, future_model, future_theta_list,
                                                          mpccal_progress=progress,
                                                          nmajor=5,
                                                          context='2d',
                                                          algorithm='hogbom',
                                                          scales=[0, 3, 10],
                                                          fractional_threshold=0.3, threshold=0.2,
                                                          gain=0.1, niter=1000, psf_support=256)
    
        (self.theta_list, residual) = arlexecute.compute(result, sync=True)
    
        combined_model = calculate_skymodel_equivalent_image(self.theta_list)
    
        psf_obs = invert_list_arlexecute_workflow([self.all_skymodel_noniso_vis], [model], context='2d', dopsf=True)
        result = restore_list_arlexecute_workflow([combined_model], psf_obs, [(residual, 0.0)])
        result = arlexecute.compute(result, sync=True)
    
        if self.persist: export_image_to_fits(residual, arl_path('test_results/test_mpccal_residual.fits'))
        if self.persist: export_image_to_fits(result[0], arl_path('test_results/test_mpccal_restored.fits'))
        if self.persist: export_image_to_fits(combined_model, arl_path('test_results/test_mpccal_deconvolved.fits'))
    
        recovered_mpccal_components = find_skycomponents(result[0], fwhm=2, threshold=0.32, npixels=12)
    
        def max_flux(elem):
            return numpy.max(elem.flux)
    
        recovered_mpccal_components = sorted(recovered_mpccal_components, key=max_flux, reverse=True)
    
        assert recovered_mpccal_components[0].name == 'Segment 9', recovered_mpccal_components[0].name
        assert numpy.abs(recovered_mpccal_components[0].flux[0, 0] - 7.7246326327167365) < 1e-7, \
            recovered_mpccal_components[0].flux[0, 0]
    
        newscreen = create_empty_image_like(self.screen)
        gaintables = [th.gaintable for th in self.theta_list]
        newscreen, weights = grid_gaintable_to_screen(self.all_skymodel_noniso_blockvis, gaintables, newscreen)
        if self.persist: export_image_to_fits(newscreen, arl_path('test_results/test_mpccal_screen.fits'))
        if self.persist: export_image_to_fits(weights, arl_path('test_results/test_mpccal_screenweights.fits'))
    
        arlexecute.close()

    def test_mpccal_MPCCAL_manysources_no_edge(self):
    
        self.actualSetup()
    
        model = create_empty_image_like(self.theta_list[0].image)
    
        if arlexecute.using_dask:
            progress = None
        else:
            progress = self.progress
    
        future_vis = arlexecute.scatter(self.all_skymodel_noniso_vis)
        future_model = arlexecute.scatter(model)
        future_theta_list = arlexecute.scatter(self.theta_list)
        result = mpccal_skymodel_list_arlexecute_workflow(future_vis, future_model, future_theta_list,
                                                          mpccal_progress=progress, window='no_edge',
                                                          nmajor=5,
                                                          context='2d',
                                                          algorithm='hogbom',
                                                          scales=[0, 3, 10],
                                                          fractional_threshold=0.3, threshold=0.2,
                                                          gain=0.1, niter=1000, psf_support=256)
    
        (self.theta_list, residual) = arlexecute.compute(result, sync=True)
    
        combined_model = calculate_skymodel_equivalent_image(self.theta_list)
    
        psf_obs = invert_list_arlexecute_workflow([self.all_skymodel_noniso_vis], [model], context='2d', dopsf=True)
        result = restore_list_arlexecute_workflow([combined_model], psf_obs, [(residual, 0.0)])
        result = arlexecute.compute(result, sync=True)
    
        if self.persist: export_image_to_fits(residual, arl_path('test_results/test_mpccal_no_edge_residual.fits'))
        if self.persist: export_image_to_fits(result[0], arl_path('test_results/test_mpccal_no_edge_restored.fits'))
        if self.persist: export_image_to_fits(combined_model, arl_path('test_results/test_mpccal_no_edge_deconvolved.fits'))
    
        recovered_mpccal_components = find_skycomponents(result[0], fwhm=2, threshold=0.32, npixels=12)
    
        def max_flux(elem):
            return numpy.max(elem.flux)
    
        recovered_mpccal_components = sorted(recovered_mpccal_components, key=max_flux, reverse=True)
    
        assert recovered_mpccal_components[0].name == 'Segment 9', recovered_mpccal_components[0].name
        assert numpy.abs(recovered_mpccal_components[0].flux[0, 0] - 7.724632632716737) < 1e-7, \
            recovered_mpccal_components[0].flux[0, 0]
    
        newscreen = create_empty_image_like(self.screen)
        gaintables = [th.gaintable for th in self.theta_list]
        newscreen, weights = grid_gaintable_to_screen(self.all_skymodel_noniso_blockvis, gaintables, newscreen)
        if self.persist: export_image_to_fits(newscreen, arl_path('test_results/test_mpccal_no_edge_screen.fits'))
        if self.persist: export_image_to_fits(weights, arl_path('test_results/test_mpccal_no_edge_screenweights.fits'))
    
        arlexecute.close()

    def test_mpccal_MPCCAL_manysources_subimages(self):
    
        self.actualSetup()
    
        model = create_empty_image_like(self.theta_list[0].image)
    
        if arlexecute.using_dask:
            progress = None
        else:
            progress = self.progress
    
        future_vis = arlexecute.scatter(self.all_skymodel_noniso_vis)
        future_model = arlexecute.scatter(model)
        future_theta_list = arlexecute.scatter(self.theta_list)
        result = mpccal_skymodel_list_arlexecute_workflow(future_vis, future_model, future_theta_list,
                                                          mpccal_progress=progress,
                                                          nmajor=5,
                                                          context='2d',
                                                          algorithm='hogbom',
                                                          scales=[0, 3, 10],
                                                          fractional_threshold=0.3, threshold=0.2,
                                                          gain=0.1, niter=1000, psf_support=256,
                                                          deconvolve_facets=8, deconvolve_overlap=8,
                                                          deconvolve_taper='tukey')
    
        (self.theta_list, residual) = arlexecute.compute(result, sync=True)
    
        combined_model = calculate_skymodel_equivalent_image(self.theta_list)
    
        psf_obs = invert_list_arlexecute_workflow([self.all_skymodel_noniso_vis], [model], context='2d', dopsf=True)
        result = restore_list_arlexecute_workflow([combined_model], psf_obs, [(residual, 0.0)])
        result = arlexecute.compute(result, sync=True)
    
        if self.persist: export_image_to_fits(residual, arl_path('test_results/test_mpccal_no_edge_residual.fits'))
        if self.persist: export_image_to_fits(result[0], arl_path('test_results/test_mpccal_no_edge_restored.fits'))
        if self.persist: export_image_to_fits(combined_model, arl_path('test_results/test_mpccal_no_edge_deconvolved.fits'))
    
        recovered_mpccal_components = find_skycomponents(result[0], fwhm=2, threshold=0.32, npixels=12)
    
        def max_flux(elem):
            return numpy.max(elem.flux)
    
        recovered_mpccal_components = sorted(recovered_mpccal_components, key=max_flux, reverse=True)
    
        assert recovered_mpccal_components[0].name == 'Segment 7', recovered_mpccal_components[0].name
        assert numpy.abs(recovered_mpccal_components[0].flux[0, 0] - 7.7580449302966725) < 1e-7, \
            recovered_mpccal_components[0].flux[0, 0]
    
        newscreen = create_empty_image_like(self.screen)
        gaintables = [th.gaintable for th in self.theta_list]
        newscreen, weights = grid_gaintable_to_screen(self.all_skymodel_noniso_blockvis, gaintables, newscreen)
        if self.persist: export_image_to_fits(newscreen, arl_path('test_results/test_mpccal_no_edge_screen.fits'))
        if self.persist: export_image_to_fits(weights, arl_path('test_results/test_mpccal_no_edge_screenweights.fits'))
    
        arlexecute.close()
