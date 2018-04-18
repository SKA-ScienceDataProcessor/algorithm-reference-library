""" Unit tests for pipelines expressed via dask.delayed


"""

import logging
import os
import sys
import unittest

import dask
import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from dask import delayed

from arl.data.polarisation import PolarisationFrame
from arl.graphs.delayed import create_zero_vis_graph_list, create_predict_graph, create_invert_graph, \
    create_subtract_vis_graph_list, create_weight_vis_graph_list
from arl.image.operations import export_image_to_fits, smooth_image, copy_image
from arl.imaging import predict_skycomponent_visibility
from arl.imaging.imaging_context import invert_function
from arl.skycomponent.operations import find_skycomponents, find_nearest_component, insert_skycomponent
from arl.util.testing_support import create_named_configuration, ingest_unittest_visibility, create_unittest_model, \
    insert_unittest_errors, create_unittest_components

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImaging(unittest.TestCase):
    def setUp(self):
        import dask.multiprocessing
        dask.set_options(get=dask.multiprocessing.get)
    
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
    
    def actualSetUp(self, add_errors=False, freqwin=3, block=False, dospectral=True, dopol=False, zerow=False):
        
        self.npixel = 256
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = freqwin
        self.vis_graph_list = list()
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        
        if freqwin > 1:
            self.channelwidth = numpy.array(freqwin * [self.frequency[1] - self.frequency[0]])
        else:
            self.channelwidth = numpy.array([1e6])
        
        if dopol:
            self.vis_pol = PolarisationFrame('linear')
            self.image_pol = PolarisationFrame('stokesIQUV')
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        else:
            self.vis_pol = PolarisationFrame('stokesI')
            self.image_pol = PolarisationFrame('stokesI')
            f = numpy.array([100.0])
        
        if dospectral:
            flux = numpy.array([f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency])
        else:
            flux = numpy.array([f])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis_graph_list = [delayed(ingest_unittest_visibility)(self.low,
                                                                   [self.frequency[freqwin]],
                                                                   [self.channelwidth[freqwin]],
                                                                   self.times,
                                                                   self.vis_pol,
                                                                   self.phasecentre, block=block,
                                                                   zerow=zerow)
                               for freqwin, _ in enumerate(self.frequency)]
        
        self.model_graph = [delayed(create_unittest_model, nout=freqwin)(self.vis_graph_list[freqwin],
                                                                         self.image_pol,
                                                                         npixel=self.npixel)
                            for freqwin, _ in enumerate(self.frequency)]
        
        self.components_graph = [delayed(create_unittest_components)(self.model_graph[freqwin],
                                                                     flux[freqwin, :][numpy.newaxis, :])
                                 for freqwin, _ in enumerate(self.frequency)]
        
        self.model_graph = [delayed(insert_skycomponent, nout=1)(self.model_graph[freqwin],
                                                                 self.components_graph[freqwin])
                            for freqwin, _ in enumerate(self.frequency)]
        
        self.vis_graph_list = [delayed(predict_skycomponent_visibility)(self.vis_graph_list[freqwin],
                                                                        self.components_graph[freqwin])
                               for freqwin, _ in enumerate(self.frequency)]
        
        # Calculate the model convolved with a Gaussian.
        self.model = self.model_graph[0].compute()
        self.cmodel = smooth_image(self.model)
        export_image_to_fits(self.model, '%s/test_imaging_model.fits' % self.dir)
        export_image_to_fits(self.cmodel, '%s/test_imaging_cmodel.fits' % self.dir)
        
        if add_errors and block:
            self.vis_graph_list = [delayed(insert_unittest_errors)(self.vis_graph_list[i])
                                   for i, _ in enumerate(self.frequency)]
        
        self.vis = self.vis_graph_list[0].compute()
        
        self.components = self.components_graph[0].compute()
    
    def test_time_setup(self):
        self.actualSetUp()
    
    def _checkcomponents(self, dirty, fluxthreshold=0.6, positionthreshold=1.0):
        comps = find_skycomponents(dirty, fwhm=1.0, threshold=10 * fluxthreshold, npixels=5)
        assert len(comps) == len(self.components), "Different number of components found: original %d, recovered %d" % \
                                                   (len(self.components), len(comps))
        cellsize = abs(dirty.wcs.wcs.cdelt[0])
        
        for comp in comps:
            # Check for agreement in direction
            ocomp = find_nearest_component(comp.direction, self.components)
            radiff = abs(comp.direction.ra.deg - ocomp.direction.ra.deg) / cellsize
            assert radiff < positionthreshold, "Component differs in dec %.3f pixels" % radiff
            decdiff = abs(comp.direction.dec.deg - ocomp.direction.dec.deg) / cellsize
            assert decdiff < positionthreshold, "Component differs in dec %.3f pixels" % decdiff
    
    def _predict_base(self, context='2d', extra='', fluxthreshold=1.0, facets=1, vis_slices=1, **kwargs):
        vis_graph_list = create_zero_vis_graph_list(self.vis_graph_list)
        vis_graph_list = create_predict_graph(vis_graph_list, self.model_graph, context=context,
                                              vis_slices=vis_slices, facets=facets, **kwargs)
        vis_graph_list = create_subtract_vis_graph_list(self.vis_graph_list, vis_graph_list)
        
        result = vis_graph_list[0].compute()
        dirty_graph = create_invert_graph([result], [self.model_graph[0]], context='2d', dopsf=False,
                                          normalize=True)
        dirty_g = dirty_graph[0].compute()[0]
        assert numpy.max(numpy.abs(dirty_g.data)), "Residual image is empty"
        export_image_to_fits(dirty_g, '%s/test_imaging_predict_%s%s_delayed_dirty.fits' %
                             (self.dir, context, extra))
        
        export_image_to_fits(dirty_g, '%s/test_imaging_predict_%s%s_delayed_dirty.fits' %
                             (self.dir, context, extra))
        maxabs = numpy.max(numpy.abs(dirty_g.data))
        assert maxabs < fluxthreshold, "Graph %s, abs max %f exceeds flux threshold" % (context, maxabs)
    
    def _invert_base(self, context, extra='', fluxthreshold=1.0, positionthreshold=1.0, check_components=True,
                     facets=1, vis_slices=1, **kwargs):
        
        dirty_graph = create_invert_graph(self.vis_graph_list, self.model_graph, context=context,
                                          dopsf=False, normalize=True, facets=facets, vis_slices=vis_slices,
                                          **kwargs)
        dirty_d = dirty_graph[0].compute()[0]
        export_image_to_fits(dirty_d, '%s/test_imaging_invert_%s%s_delayed_dirty.fits' %
                             (self.dir, context, extra))
        
        dirty_f = invert_function(self.vis, self.model, context=context,
                                  dopsf=False, normalize=True, facets=facets, vis_slices=vis_slices,
                                  **kwargs)[0]
        export_image_to_fits(dirty_d, '%s/test_imaging_invert_%s%s_function_dirty.fits' %
                             (self.dir, context, extra))
        
        assert numpy.max(numpy.abs(dirty_f.data)), "Function image is empty"
        assert numpy.max(numpy.abs(dirty_d.data)), "Delayed image is empty"

        difference = copy_image(dirty_d)
        difference.data -= dirty_f.data
        maxabs = numpy.max(numpy.abs(difference.data))
        if maxabs > 1e-8:
            export_image_to_fits(difference, '%s/test_imaging_invert_%s%s_difference_dirty.fits' %
                                 (self.dir, context, extra))
        
        assert maxabs < 1e-8, "Difference between delayed and function for %s, abs max %s is non-zero " \
                              % (context, str(maxabs))
        
        if check_components:
            self._checkcomponents(dirty_d, fluxthreshold, positionthreshold)
            self._checkcomponents(dirty_f, fluxthreshold, positionthreshold)
    
    def test_predict_2d(self):
        self.actualSetUp(zerow=True)
        self._predict_base(context='2d')
    
    @unittest.skip("Facets invert requires overlap")
    def test_predict_facets(self):
        self.actualSetUp()
        self._predict_base(context='facets', fluxthreshold=15.0, facets=8)
    
    @unittest.skip("Timeslice predict needs better interpolation")
    def test_predict_facets_timeslice(self):
        self.actualSetUp()
        self._predict_base(context='facets_timeslice', fluxthreshold=19.0, facets=8, vis_slices=self.ntimes)
    
    @unittest.skip("Facets invert requires overlap")
    def test_predict_facets_wprojection(self):
        self.actualSetUp()
        self._predict_base(context='facets', extra='_wprojection', facets=8, wstep=8.0, fluxthreshold=15.0)
    
    @unittest.skip("Correcting twice?")
    def test_predict_facets_wstack(self):
        self.actualSetUp()
        self._predict_base(context='facets_wstack', fluxthreshold=15.0, facets=8, vis_slices=41)
    
    @unittest.skip("Timeslice predict needs better interpolation")
    def test_predict_timeslice(self):
        self.actualSetUp()
        self._predict_base(context='timeslice', fluxthreshold=19.0, vis_slices=self.ntimes)
    
    @unittest.skip("Timeslice predict needs better interpolation")
    def test_predict_timeslice_wprojection(self):
        self.actualSetUp()
        self._predict_base(context='timeslice', extra='_wprojection', fluxthreshold=3.0, wstep=10.0,
                           vis_slices=self.ntimes)
    
    def test_predict_wprojection(self):
        self.actualSetUp()
        self._predict_base(context='2d', extra='_wprojection', wstep=10.0, fluxthreshold=1.0)
    
    def test_predict_wstack(self):
        self.actualSetUp()
        self._predict_base(context='wstack', fluxthreshold=2.0, vis_slices=41)
    
    def test_predict_wstack_wprojection(self):
        self.actualSetUp()
        self._predict_base(context='wstack', extra='_wprojection', fluxthreshold=3.0, wstep=10.0, vis_slices=41)
    
    def test_predict_wstack_spectral(self):
        self.actualSetUp(dospectral=True)
        self._predict_base(context='wstack', extra='_spectral', fluxthreshold=4.0, vis_slices=41)
    
    def test_predict_wstack_spectral_pol(self):
        self.actualSetUp(dospectral=True, dopol=True)
        self._predict_base(context='wstack', extra='_spectral', fluxthreshold=4.0, vis_slices=41)
    
    def test_invert_2d(self):
        self.actualSetUp(zerow=True)
        self._invert_base(context='2d', positionthreshold=2.0, check_components=False)
    
    def test_invert_facets(self):
        self.actualSetUp()
        self._invert_base(context='facets', positionthreshold=2.0, check_components=True, facets=8)
    
    @unittest.skip("Correcting twice?")
    def test_invert_facets_timeslice(self):
        self.actualSetUp()
        self._invert_base(context='facets_timeslice', check_components=True, vis_slices=self.ntimes,
                          positionthreshold=5.0, flux_threshold=1.0, facets=8)
    
    def test_invert_facets_wprojection(self):
        self.actualSetUp()
        self._invert_base(context='facets', extra='_wprojection', check_components=True,
                          positionthreshold=2.0, wstep=10.0)
    
    @unittest.skip("Correcting twice?")
    def test_invert_facets_wstack(self):
        self.actualSetUp()
        self._invert_base(context='facets_wstack', positionthreshold=1.0, check_components=False, facets=8,
                          vis_slices=41)
    
    def test_invert_timeslice(self):
        self.actualSetUp()
        self._invert_base(context='timeslice', positionthreshold=1.0, check_components=True,
                          vis_slices=self.ntimes)
    
    def test_invert_timeslice_wprojection(self):
        self.actualSetUp()
        self._invert_base(context='timeslice', extra='_wprojection', positionthreshold=1.0,
                          check_components=True, wstep=20.0, vis_slices=self.ntimes)
    
    def test_invert_wprojection(self):
        self.actualSetUp()
        self._invert_base(context='2d', extra='_wprojection', positionthreshold=2.0, wstep=10.0)
    
    def test_invert_wprojection_wstack(self):
        self.actualSetUp()
        self._invert_base(context='wstack', extra='_wprojection', positionthreshold=1.0, wstep=10.0,
                          vis_slices=11)
    
    def test_invert_wstack(self):
        self.actualSetUp()
        self._invert_base(context='wstack', positionthreshold=1.0, vis_slices=41)
    
    def test_invert_wstack_spectral(self):
        self.actualSetUp(dospectral=True)
        self._invert_base(context='wstack', extra='_spectral', positionthreshold=2.0,
                          vis_slices=41)
    
    def test_invert_wstack_spectral_pol(self):
        self.actualSetUp(dospectral=True, dopol=True)
        self._invert_base(context='wstack', extra='_spectral_pol', positionthreshold=2.0,
                          vis_slices=41)
    
    def test_weighting(self):
        
        self.actualSetUp()

        context = 'wstack'
        vis_slices = 41
        facets = 1
        
        dirty_graph = create_invert_graph(self.vis_graph_list, self.model_graph, context=context,
                                          dopsf=False, normalize=True, facets=facets, vis_slices=vis_slices)
        dirty_g = dirty_graph[0].compute()[0]
        export_image_to_fits(dirty_g, '%s/test_imaging_noweighting_delayed_dirty.fits' % self.dir)
        
        self.vis_graph_list = create_weight_vis_graph_list(self.vis_graph_list, self.model_graph, weighting='uniform')
        
        dirty_graph = create_invert_graph(self.vis_graph_list, self.model_graph, context=context,
                                          dopsf=False, normalize=True, facets=facets, vis_slices=vis_slices)
        dirty_f = dirty_graph[0].compute()[0]
        export_image_to_fits(dirty_f, '%s/test_imaging_weighting_delayed_dirty.fits'% self.dir)


if __name__ == '__main__':
    unittest.main()
