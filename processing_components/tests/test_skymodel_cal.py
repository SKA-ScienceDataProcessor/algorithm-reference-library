""" Unit tests for calibration solution


"""
import logging
import os
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from libs.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility
from libs.calibration.solvers import solve_gaintable
from data_models.polarisation import PolarisationFrame
from data_models.data_models import SkyModel
from libs.image.operations import qa_image, export_image_to_fits
from libs.imaging import predict_skycomponent_visibility, create_image_from_visibility
from libs.imaging.imaging_functions import invert_function
from libs.imaging.weighting import weight_visibility
from libs.skycomponent.operations import apply_beam_to_skycomponent
from libs.util.testing_support import create_named_configuration, simulate_gaintable, \
    create_low_test_skycomponents_from_gleam, create_low_test_beam
from libs.visibility.base import copy_visibility, create_blockvisibility
from libs.visibility.coalesce import convert_blockvisibility_to_visibility

from processing_components.graphs.execute import arlexecute
from processing_components.graphs.skymodel_cal_graph import create_skymodel_cal_solve_graph

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
    
    def test_skymodel_cal_solve_graph(self):
        
        self.actualSetup(doiso=True)
        
        self.skymodel_graph = [arlexecute.execute(SkyModel, nout=1)(components=[cm]) for cm in self.components]
        
        skymodel_cal_graph = create_skymodel_cal_solve_graph(self.vis, self.skymodel_graph, niter=30,
                                                             gain=0.25, tol=1e-8)
        skymodel, residual_vis = arlexecute.compute(skymodel_cal_graph)
        
        residual_vis = convert_blockvisibility_to_visibility(residual_vis)
        residual_vis, _, _ = weight_visibility(residual_vis, self.beam)
        dirty, sumwt = invert_function(residual_vis, self.beam, context='2d')
        export_image_to_fits(dirty, "%s/test_skymodel_cal-%s-final-iso-residual.fits" %
                             (self.dir, arlexecute.type()))
        
        qa = qa_image(dirty)
        assert qa.data['rms'] < 3.2e-3, qa


if __name__ == '__main__':
    unittest.main()
