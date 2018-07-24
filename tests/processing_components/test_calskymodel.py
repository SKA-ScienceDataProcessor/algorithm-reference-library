""" Unit tests for calibration solution


"""
import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from data_models.memory_data_models import SkyModel

from processing_components.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility
from processing_components.calibration.modelpartition import modelpartition_solve
from processing_components.calibration.calibration import solve_gaintable
from processing_components.image.operations import export_image_to_fits, qa_image
from processing_components.imaging.base import predict_skycomponent_visibility, create_image_from_visibility
from processing_components.imaging.imaging_functions import invert_function
from processing_components.imaging.weighting import weight_visibility
from processing_components.skycomponent.operations import apply_beam_to_skycomponent, find_skycomponent_matches
from processing_components.simulation.testing_support import create_named_configuration, simulate_gaintable, \
    create_low_test_skycomponents_from_gleam, create_low_test_beam
from processing_components.visibility.base import copy_visibility, create_blockvisibility
from processing_components.visibility.coalesce import convert_blockvisibility_to_visibility

log = logging.getLogger(__name__)


class TestCalibrationSkyModelcal(unittest.TestCase):
    def setUp(self):
        
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        numpy.random.seed(180555)


    def actualSetup(self, vnchan=1, doiso=True, ntimes=5, flux_limit=2.0, zerow=True, fixed=False):
        
        nfreqwin = vnchan
        rmax = 300.0
        npixel = 512
        cellsize = 0.001
        frequency = numpy.linspace(0.8e8, 1.2e8, nfreqwin)
        if nfreqwin > 1:
            channel_bandwidth = numpy.array(nfreqwin * [frequency[1] - frequency[0]])
        else:
            channel_bandwidth = [0.4e8]
        times = numpy.linspace(-numpy.pi / 3.0, numpy.pi / 3.0, ntimes)
        
        phasecentre = SkyCoord(ra=-60.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        
        lowcore = create_named_configuration('LOWBD2', rmax=rmax)
        
        block_vis = create_blockvisibility(lowcore, times, frequency=frequency, channel_bandwidth=channel_bandwidth,
                                           weight=1.0, phasecentre=phasecentre,
                                           polarisation_frame=PolarisationFrame("stokesI"), zerow=zerow)
        
        block_vis.data['uvw'][..., 2] = 0.0
        self.beam = create_image_from_visibility(block_vis, npixel=npixel, frequency=[numpy.average(frequency)],
                                                 nchan=nfreqwin,
                                                 channel_bandwidth=[numpy.sum(channel_bandwidth)], cellsize=cellsize,
                                                 phasecentre=phasecentre)
        
        self.components = create_low_test_skycomponents_from_gleam(flux_limit=flux_limit, phasecentre=phasecentre,
                                                                   frequency=frequency,
                                                                   polarisation_frame=PolarisationFrame('stokesI'),
                                                                   radius=npixel * cellsize)
        self.beam = create_low_test_beam(self.beam)
        self.components = apply_beam_to_skycomponent(self.components, self.beam, flux_limit=flux_limit)
        
        self.vis = copy_visibility(block_vis, zero=True)
        gt = create_gaintable_from_blockvisibility(block_vis, timeslice='auto')
        for i, sc in enumerate(self.components):
            if sc.flux[0, 0] > 10:
                sc.flux[...] /= 10.0
            component_vis = copy_visibility(block_vis, zero=True)
            gt = simulate_gaintable(gt, amplitude_error=0.0, phase_error=0.1, seed=None)
            component_vis = predict_skycomponent_visibility(component_vis, sc)
            component_vis = apply_gaintable(component_vis, gt)
            self.vis.data['vis'][...] += component_vis.data['vis'][...]
        
        # Do an isoplanatic selfcal
        self.model_vis = copy_visibility(self.vis, zero=True)
        self.model_vis = predict_skycomponent_visibility(self.model_vis, self.components)
        if doiso:
            gt = solve_gaintable(self.vis, self.model_vis, phase_only=True, timeslice='auto')
            self.vis = apply_gaintable(self.vis, gt, inverse=True)
        
        self.model_vis = convert_blockvisibility_to_visibility(self.model_vis)
        self.model_vis, _, _ = weight_visibility(self.model_vis, self.beam)
        self.dirty_model, sumwt = invert_function(self.model_vis, self.beam, context='2d')
        export_image_to_fits(self.dirty_model, "%s/test_skymodel-model_dirty.fits" % self.dir)
        
        lvis = convert_blockvisibility_to_visibility(self.vis)
        lvis, _, _ = weight_visibility(lvis, self.beam)
        dirty, sumwt = invert_function(lvis, self.beam, context='2d')
        if doiso:
            export_image_to_fits(dirty, "%s/test_skymodel-initial-iso-residual.fits" % self.dir)
        else:
            export_image_to_fits(dirty, "%s/test_skymodel-initial-noiso-residual.fits" % self.dir)
        
        self.skymodels = [SkyModel(components=[cm], fixed=fixed) for cm in self.components]
    
    def test_time_setup(self):
        self.actualSetup()
    
    def test_skymodel_solve(self):
        self.actualSetup(ntimes=1, doiso=True)
        modelpartition, residual_vis = modelpartition_solve(self.vis, self.skymodels, niter=30, gain=0.25)
        
        residual_vis = convert_blockvisibility_to_visibility(residual_vis)
        residual_vis, _, _ = weight_visibility(residual_vis, self.beam)
        dirty, sumwt = invert_function(residual_vis, self.beam, context='2d')
        export_image_to_fits(dirty, "%s/test_skymodel-final-iso-residual.fits" % self.dir)
        
        qa = qa_image(dirty)
        assert qa.data['rms'] < 3.4e-3, qa
    
    def test_skymodel_solve_fixed(self):
        self.actualSetup(ntimes=1, doiso=True, fixed=True)
        modelpartition, residual_vis = modelpartition_solve(self.vis, self.skymodels, niter=30, gain=0.25)
        
        # Check that the components are unchanged
        modelpartition_skycomponents = list()
        for sm in [csm[0] for csm in modelpartition]:
            for comp in sm.components:
                modelpartition_skycomponents.append(comp)
        
        recovered_components = find_skycomponent_matches(modelpartition_skycomponents, self.components, 1e-5)
        for p in recovered_components:
            assert numpy.abs(modelpartition_skycomponents[p[0]].flux[0, 0] - self.components[p[1]].flux[0, 0]) < 1e-15
            assert modelpartition_skycomponents[p[0]].direction.separation(self.components[p[1]].direction).rad < 1e-15
        
        residual_vis = convert_blockvisibility_to_visibility(residual_vis)
        residual_vis, _, _ = weight_visibility(residual_vis, self.beam)
        dirty, sumwt = invert_function(residual_vis, self.beam, context='2d')
        export_image_to_fits(dirty, "%s/test_skymodel-final-iso-residual.fits" % self.dir)
        
        qa = qa_image(dirty)
        assert qa.data['rms'] < 3.4e-3, qa
    
    def test_skymodel_solve_noiso(self):
        self.actualSetup(ntimes=1, doiso=False)
        modelpartition, residual_vis = modelpartition_solve(self.vis, self.skymodels, niter=30, gain=0.25)
        
        residual_vis = convert_blockvisibility_to_visibility(residual_vis)
        residual_vis, _, _ = weight_visibility(residual_vis, self.beam)
        dirty, sumwt = invert_function(residual_vis, self.beam, context='2d')
        export_image_to_fits(dirty, "%s/test_skymodel-final-noiso-residual.fits" % self.dir)
        
        qa = qa_image(dirty)
        assert qa.data['rms'] < 3.8e-3, qa
    
if __name__ == '__main__':
    unittest.main()
