""" Unit tests for pipelines expressed via arlexecute
"""

import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from data_models.memory_data_models import BlockVisibility, Visibility
from processing_components.griddata import apply_bounding_box_convolutionfunction
from processing_components.griddata.kernels import create_awterm_convolutionfunction
from workflows.arlexecute.imaging.imaging_arlexecute import zero_list_arlexecute_workflow, \
    predict_list_arlexecute_workflow, invert_list_arlexecute_workflow, subtract_list_arlexecute_workflow, \
    weight_list_arlexecute_workflow, residual_list_arlexecute_workflow, sum_invert_results_arlexecute, \
    restore_list_arlexecute_workflow
from workflows.shared.imaging.imaging_shared import sum_invert_results, sum_invert_results_local
from wrappers.arlexecute.execution_support.arlexecutebase import ARLExecuteBase
from wrappers.arlexecute.execution_support.dask_init import get_dask_Client

from processing_components.image.operations import export_image_to_fits, smooth_image, qa_image
from processing_components.imaging.base import predict_skycomponent_visibility
from processing_components.simulation import ingest_unittest_visibility, \
    create_unittest_model, insert_unittest_errors, create_unittest_components
from processing_components.simulation import create_named_configuration
from processing_components.skycomponent.operations import find_skycomponents, find_nearest_skycomponent, \
    insert_skycomponent

from processing_components.visibility.coalesce import convert_blockvisibility_to_visibility

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImaging(unittest.TestCase):
    def setUp(self):
        
        client = get_dask_Client(memory_limit=4 * 1024 * 1024 * 1024, n_workers=4, dashboard_address=None)
        global arlexecute
        arlexecute = ARLExecuteBase(use_dask=True)
        arlexecute.set_client(client, verbose=True)

        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
    
        self.persist = False
    
    def tearDown(self):
        global arlexecute
        arlexecute.close()
        del arlexecute

    def actualSetUp(self, add_errors=False, freqwin=3, block=False, dospectral=True, dopol=False, zerow=False,
                    makegcfcf=False):
        
        self.npixel = 256
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = freqwin
        self.vis_list = list()
        self.ntimes = 5
        self.cellsize = 0.0005
        # Choose the interval so that the maximum change in w is smallish
        integration_time = numpy.pi * (24 / (12 * 60))
        self.times = numpy.linspace(-integration_time * (self.ntimes // 2), integration_time * (self.ntimes // 2),
                                    self.ntimes)
        
        if freqwin > 1:
            self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
            self.channelwidth = numpy.array(freqwin * [self.frequency[1] - self.frequency[0]])
        else:
            self.frequency = numpy.array([1.0e8])
            self.channelwidth = numpy.array([4e7])
        
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
        self.bvis_list = [arlexecute.execute(ingest_unittest_visibility)(self.low,
                                                                        [self.frequency[freqwin]],
                                                                        [self.channelwidth[freqwin]],
                                                                        self.times,
                                                                        self.vis_pol,
                                                                        self.phasecentre, block=True,
                                                                        zerow=zerow)
                         for freqwin, _ in enumerate(self.frequency)]
        self.vis_list = [arlexecute.execute(convert_blockvisibility_to_visibility)(bvis) for bvis in self.bvis_list]
        
        self.model_list = [arlexecute.execute(create_unittest_model, nout=freqwin)(self.vis_list[freqwin],
                                                                                   self.image_pol,
                                                                                   cellsize=self.cellsize,
                                                                                   npixel=self.npixel)
                           for freqwin, _ in enumerate(self.frequency)]
        
        self.components_list = [arlexecute.execute(create_unittest_components)(self.model_list[freqwin],
                                                                               flux[freqwin, :][numpy.newaxis, :],
                                                                               single=True)
                                for freqwin, _ in enumerate(self.frequency)]
        
        self.components_list = arlexecute.compute(self.components_list, sync=True)
        
        self.model_list = [arlexecute.execute(insert_skycomponent, nout=1)(self.model_list[freqwin],
                                                                           self.components_list[freqwin])
                           for freqwin, _ in enumerate(self.frequency)]
        
        self.model_list = arlexecute.compute(self.model_list, sync=True)
        
        self.vis_list = [arlexecute.execute(predict_skycomponent_visibility)(self.vis_list[freqwin],
                                                                             self.components_list[freqwin])
                         for freqwin, _ in enumerate(self.frequency)]
        centre = self.freqwin // 2
        # Calculate the model convolved with a Gaussian.
        self.model = self.model_list[centre]
        
        self.cmodel = smooth_image(self.model)
        if self.persist: export_image_to_fits(self.model, '%s/test_imaging_model.fits' % self.dir)
        if self.persist: export_image_to_fits(self.cmodel, '%s/test_imaging_cmodel.fits' % self.dir)
        
        if add_errors and block:
            self.vis_list = [arlexecute.execute(insert_unittest_errors)(self.vis_list[i])
                             for i, _ in enumerate(self.frequency)]
        
        self.components = self.components_list[centre]
        
        if makegcfcf:
            self.gcfcf = [create_awterm_convolutionfunction(self.model, nw=61, wstep=16.0,
                                                            oversampling=8,
                                                            support=64,
                                                            use_aaf=True)]
            self.gcfcf_clipped = [(self.gcfcf[0][0], apply_bounding_box_convolutionfunction(self.gcfcf[0][1],
                                                                                            fractional_level=1e-3))]
            
            self.gcfcf_joint = [create_awterm_convolutionfunction(self.model, nw=11, wstep=16.0,
                                                                  oversampling=8,
                                                                  support=64,
                                                                  use_aaf=True)]
        
        else:
            self.gcfcf = None
            self.gcfcf_clipped = None
            self.gcfcf_joint = None
    
    def test_time_setup(self):
        self.actualSetUp()
    
    def _checkcomponents(self, dirty, fluxthreshold=0.6, positionthreshold=1.0):
        comps = find_skycomponents(dirty, fwhm=1.0, threshold=10 * fluxthreshold, npixels=5)
        assert len(comps) == len(self.components), "Different number of components found: original %d, recovered %d" % \
                                                   (len(self.components), len(comps))
        cellsize = abs(dirty.wcs.wcs.cdelt[0])
        
        for comp in comps:
            # Check for agreement in direction
            ocomp, separation = find_nearest_skycomponent(comp.direction, self.components)
            assert separation / cellsize < positionthreshold, "Component differs in position %.3f pixels" % \
                                                              separation / cellsize
    
    def _predict_base(self, context='2d', extra='', fluxthreshold=1.0, facets=1, vis_slices=1,
                      gcfcf=None, **kwargs):
        centre = self.freqwin // 2
        
        vis_list = zero_list_arlexecute_workflow(self.vis_list)
        vis_list = predict_list_arlexecute_workflow(vis_list, self.model_list, context=context,
                                                    vis_slices=vis_slices, facets=facets,
                                                    gcfcf=gcfcf, **kwargs)
        vis_list = subtract_list_arlexecute_workflow(self.vis_list, vis_list)
        vis_list = arlexecute.compute(vis_list, sync=True)
        
        dirty = invert_list_arlexecute_workflow(vis_list, self.model_list, context=context, dopsf=False,
                                                gcfcf=gcfcf, normalize=True, vis_slices=vis_slices)
        dirty = arlexecute.compute(dirty, sync=True)[centre]
        
        assert numpy.max(numpy.abs(dirty[0].data)), "Residual image is empty"
        if self.persist: export_image_to_fits(dirty[0], '%s/test_imaging_predict_%s%s_%s_dirty.fits' %
                             (self.dir, context, extra, arlexecute.type()))
        
        maxabs = numpy.max(numpy.abs(dirty[0].data))
        assert maxabs < fluxthreshold, "Error %.3f greater than fluxthreshold %.3f " % (maxabs, fluxthreshold)
    
    def _invert_base(self, context, extra='', fluxthreshold=1.0, positionthreshold=1.0, check_components=True,
                     facets=1, vis_slices=1, gcfcf=None, **kwargs):
        
        centre = self.freqwin // 2
        dirty = invert_list_arlexecute_workflow(self.vis_list, self.model_list, context=context,
                                                dopsf=False, normalize=True, facets=facets, vis_slices=vis_slices,
                                                gcfcf=gcfcf, **kwargs)
        dirty = arlexecute.compute(dirty, sync=True)[centre]
        
        print(dirty)
        if self.persist: export_image_to_fits(dirty[0], '%s/test_imaging_invert_%s%s_%s_dirty.fits' %
                             (self.dir, context, extra, arlexecute.type()))
        
        assert numpy.max(numpy.abs(dirty[0].data)), "Image is empty"
        
        if check_components:
            self._checkcomponents(dirty[0], fluxthreshold, positionthreshold)
    
    def test_predict_2d(self):
        self.actualSetUp(zerow=True)
        self._predict_base(context='2d')
    
    @unittest.skip("Facets need overlap")
    def test_predict_facets(self):
        self.actualSetUp()
        self._predict_base(context='facets', fluxthreshold=17.0, facets=4)
    
    @unittest.skip("Timeslice predict needs better interpolation and facets need overlap")
    def test_predict_facets_timeslice(self):
        self.actualSetUp()
        self._predict_base(context='facets_timeslice', fluxthreshold=19.0, facets=8, vis_slices=self.ntimes)
    
    @unittest.skip("Facets need overlap")
    def test_predict_facets_wprojection(self, makegcfcf=True):
        self.actualSetUp()
        self._predict_base(context='facets', extra='_wprojection', facets=8, fluxthreshold=15.0,
                           gcfcf=self.gcfcf_joint)
    
    @unittest.skip("Facets need overlap")
    def test_predict_facets_wstack(self):
        self.actualSetUp()
        self._predict_base(context='facets_wstack', fluxthreshold=15.0, facets=8, vis_slices=101)
    
    def test_predict_timeslice(self):
        self.actualSetUp()
        self._predict_base(context='timeslice', fluxthreshold=3.0, vis_slices=self.ntimes)
    
    def test_predict_wsnapshots(self):
        self.actualSetUp(makegcfcf=True)
        self._predict_base(context='wsnapshots', fluxthreshold=3.0,
                           vis_slices=self.ntimes // 2, gcfcf=self.gcfcf_joint)
    
    def test_predict_wprojection(self):
        self.actualSetUp(makegcfcf=True)
        self._predict_base(context='2d', extra='_wprojection', fluxthreshold=1.0,
                           gcfcf=self.gcfcf)
    
    def test_predict_wprojection_clip(self):
        self.actualSetUp(makegcfcf=True)
        self._predict_base(context='2d', extra='_wprojection_clipped', fluxthreshold=1.0,
                           gcfcf=self.gcfcf_clipped)
    
    def test_predict_wstack(self):
        self.actualSetUp()
        self._predict_base(context='wstack', fluxthreshold=1.0, vis_slices=101)
    
    def test_predict_wstack_serial(self):
        self.actualSetUp()
        self._predict_base(context='wstack', fluxthreshold=1.0, vis_slices=101, use_serial_predict=True)
    
    def test_predict_wstack_wprojection(self):
        self.actualSetUp(makegcfcf=True)
        self._predict_base(context='wstack', extra='_wprojection', fluxthreshold=1.0, vis_slices=11,
                           gcfcf=self.gcfcf_joint)
    
    def test_predict_wstack_spectral(self):
        self.actualSetUp(dospectral=True)
        self._predict_base(context='wstack', extra='_spectral', fluxthreshold=4.0, vis_slices=101)
    
    @unittest.skip("Too much for jenkins")
    def test_predict_wstack_spectral_pol(self):
        self.actualSetUp(dospectral=True, dopol=True)
        self._predict_base(context='wstack', extra='_spectral', fluxthreshold=4.0, vis_slices=101)
    
    def test_invert_2d(self):
        self.actualSetUp(zerow=True)
        self._invert_base(context='2d', positionthreshold=2.0, check_components=False)

    def test_invert_2d_uniform(self):
        self.actualSetUp(zerow=True, makegcfcf=True)
        self.vis_list = weight_list_arlexecute_workflow(self.vis_list, self.model_list, gcfcf=self.gcfcf,
                                                        weighting='uniform')
        self._invert_base(context='2d', extra='_uniform', positionthreshold=2.0, check_components=False)

    def test_invert_2d_uniform_block(self):
        self.actualSetUp(zerow=True, makegcfcf=True, block=True)
        self.bvis_list = weight_list_arlexecute_workflow(self.bvis_list, self.model_list, gcfcf=self.gcfcf,
                                                        weighting='uniform')
        self.bvis_list = arlexecute.compute(self.bvis_list, sync=True)
        assert isinstance(self.bvis_list[0], BlockVisibility)

    def test_invert_2d_uniform_nogcfcf(self):
        self.actualSetUp(zerow=True)
        self.vis_list = weight_list_arlexecute_workflow(self.vis_list, self.model_list)
        self._invert_base(context='2d', extra='_uniform', positionthreshold=2.0, check_components=False)
    
    @unittest.skip("Facets need overlap")
    def test_invert_facets(self):
        self.actualSetUp()
        self._invert_base(context='facets', positionthreshold=2.0, check_components=True, facets=8)
    
    @unittest.skip("Facets need overlap")
    def test_invert_facets_timeslice(self):
        self.actualSetUp()
        self._invert_base(context='facets_timeslice', check_components=True, vis_slices=self.ntimes,
                          positionthreshold=5.0, flux_threshold=1.0, facets=8)
    
    @unittest.skip("Facets need overlap")
    def test_invert_facets_wprojection(self):
        self.actualSetUp(makegcfcf=True)
        self._invert_base(context='facets', extra='_wprojection', check_components=True,
                          positionthreshold=2.0, facets=4, gcfcf=self.gcfcf)
    
    @unittest.skip("Facets need overlap")
    def test_invert_facets_wstack(self):
        self.actualSetUp()
        self._invert_base(context='facets_wstack', positionthreshold=1.0, check_components=False, facets=4,
                          vis_slices=101)
    
    def test_invert_timeslice(self):
        self.actualSetUp()
        self._invert_base(context='timeslice', positionthreshold=1.0, check_components=True,
                          vis_slices=self.ntimes)
    
    def test_invert_wsnapshots(self):
        self.actualSetUp(makegcfcf=True)
        self._invert_base(context='wsnapshots', positionthreshold=1.0,
                          check_components=True, vis_slices=self.ntimes // 2, gcfcf=self.gcfcf_joint)
    
    def test_invert_wprojection(self):
        self.actualSetUp(makegcfcf=True)
        self._invert_base(context='2d', extra='_wprojection', positionthreshold=2.0, gcfcf=self.gcfcf)
    
    def test_invert_wprojection_clip(self):
        self.actualSetUp(makegcfcf=True)
        self._invert_base(context='2d', extra='_wprojection_clipped', positionthreshold=2.0,
                          gcfcf=self.gcfcf_clipped)
    
    def test_invert_wprojection_wstack(self):
        self.actualSetUp(makegcfcf=True)
        self._invert_base(context='wstack', extra='_wprojection', positionthreshold=1.0, vis_slices=11,
                          gcfcf=self.gcfcf_joint)
    
    def test_invert_wstack(self):
        self.actualSetUp()
        self._invert_base(context='wstack', positionthreshold=1.0, vis_slices=101)
    
    def test_invert_wstack_spectral(self):
        self.actualSetUp(dospectral=True)
        self._invert_base(context='wstack', extra='_spectral', positionthreshold=2.0,
                          vis_slices=101)
    
    @unittest.skip("Too much for jenkins")
    def test_invert_wstack_spectral_pol(self):
        self.actualSetUp(dospectral=True, dopol=True)
        self._invert_base(context='wstack', extra='_spectral_pol', positionthreshold=2.0,
                          vis_slices=101)
    
    def test_zero_list(self):
        self.actualSetUp()
        
        centre = self.freqwin // 2
        vis_list = zero_list_arlexecute_workflow(self.vis_list)
        vis_list = arlexecute.compute(vis_list, sync=True)
        
        assert numpy.max(numpy.abs(vis_list[centre].vis)) < 1e-15, numpy.max(numpy.abs(vis_list[centre].vis))
        
        predicted_vis_list = [arlexecute.execute(predict_skycomponent_visibility)(vis_list[freqwin],
                                                                                  self.components_list[freqwin])
                              for freqwin, _ in enumerate(self.frequency)]
        predicted_vis_list = arlexecute.compute(predicted_vis_list, sync=True)
        assert numpy.max(numpy.abs(predicted_vis_list[centre].vis)) > 0.0, \
            numpy.max(numpy.abs(predicted_vis_list[centre].vis))
        
        diff_vis_list = subtract_list_arlexecute_workflow(self.vis_list, predicted_vis_list)
        diff_vis_list = arlexecute.compute(diff_vis_list, sync=True)
        
        assert numpy.max(numpy.abs(diff_vis_list[centre].vis)) < 1e-15, numpy.max(numpy.abs(diff_vis_list[centre].vis))

    def test_residual_list(self):
        self.actualSetUp(zerow=True)
    
        centre = self.freqwin // 2
        residual_image_list = residual_list_arlexecute_workflow(self.vis_list, self.model_list, context='2d')
        residual_image_list = arlexecute.compute(residual_image_list, sync=True)
        qa = qa_image(residual_image_list[centre][0])
        assert numpy.abs(qa.data['max'] - 0.35139716991480785) < 1.0, str(qa)
        assert numpy.abs(qa.data['min'] + 0.7681701460717593) < 1.0, str(qa)

    def test_restored_list(self):
        self.actualSetUp(zerow=True)
        
        centre = self.freqwin // 2
        psf_image_list = invert_list_arlexecute_workflow(self.vis_list, self.model_list, context='2d', dopsf=True)
        residual_image_list = residual_list_arlexecute_workflow(self.vis_list, self.model_list, context='2d')
        restored_image_list = restore_list_arlexecute_workflow(self.model_list, psf_image_list, residual_image_list,
                                                               psfwidth=1.0)
        restored_image_list = arlexecute.compute(restored_image_list, sync=True)
        if self.persist: export_image_to_fits(restored_image_list[centre], '%s/test_imaging_invert_%s_restored.fits' %
                                              (self.dir, arlexecute.type()))
        
        qa = qa_image(restored_image_list[centre])
        assert numpy.abs(qa.data['max'] - 99.43438263927834) < 1e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.6328915148563365) < 1e-7, str(qa)
    
    def test_restored_list_noresidual(self):
        self.actualSetUp(zerow=True)
        
        centre = self.freqwin // 2
        psf_image_list = invert_list_arlexecute_workflow(self.vis_list, self.model_list, context='2d', dopsf=True)
        restored_image_list = restore_list_arlexecute_workflow(self.model_list, psf_image_list, psfwidth=1.0)
        restored_image_list = arlexecute.compute(restored_image_list, sync=True)
        if self.persist: export_image_to_fits(restored_image_list[centre],
                                              '%s/test_imaging_invert_%s_restored_noresidual.fits' %
                                              (self.dir, arlexecute.type()))
        
        qa = qa_image(restored_image_list[centre])
        assert numpy.abs(qa.data['max'] - 100.0) < 1e-7, str(qa)
        assert numpy.abs(qa.data['min']) < 1e-7, str(qa)
    
    def test_restored_list_facet(self):
        self.actualSetUp(zerow=True)
        
        centre = self.freqwin // 2
        psf_image_list = invert_list_arlexecute_workflow(self.vis_list, self.model_list, context='2d', dopsf=True)
        residual_image_list = residual_list_arlexecute_workflow(self.vis_list, self.model_list, context='2d')
        restored_4facets_image_list = restore_list_arlexecute_workflow(self.model_list, psf_image_list,
                                                                       residual_image_list,
                                                                       restore_facets=4, psfwidth=1.0)
        restored_4facets_image_list = arlexecute.compute(restored_4facets_image_list, sync=True)
        
        restored_1facets_image_list = restore_list_arlexecute_workflow(self.model_list, psf_image_list,
                                                                       residual_image_list,
                                                                       restore_facets=1, psfwidth=1.0)
        restored_1facets_image_list = arlexecute.compute(restored_1facets_image_list, sync=True)
        
        if self.persist: export_image_to_fits(restored_4facets_image_list[0],
                                              '%s/test_imaging_invert_%s_restored_4facets.fits' %
                                              (self.dir, arlexecute.type()))
        
        qa = qa_image(restored_4facets_image_list[centre])
        assert numpy.abs(qa.data['max'] - 99.43438263927833) < 1e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.6328915148563354) < 1e-7, str(qa)
        
        restored_4facets_image_list[centre].data -= restored_1facets_image_list[centre].data
        if self.persist: export_image_to_fits(restored_4facets_image_list[centre],
                                              '%s/test_imaging_invert_%s_restored_4facets_error.fits' %
                                              (self.dir, arlexecute.type()))
        qa = qa_image(restored_4facets_image_list[centre])
        assert numpy.abs(qa.data['max']) < 1e-10, str(qa)
    
    def test_sum_invert_list(self):
        self.actualSetUp(zerow=True)
    
        residual_image_list = residual_list_arlexecute_workflow(self.vis_list, self.model_list, context='2d')
        residual_image_list = arlexecute.compute(residual_image_list, sync=True)
        route2 = sum_invert_results(residual_image_list)
        route1 = sum_invert_results_arlexecute(residual_image_list)
        route1 = arlexecute.compute(route1, sync=True)
        for r in route1, route2:
            assert len(r) == 2
            qa = qa_image(r[0])
            assert numpy.abs(qa.data['max'] - 0.35139716991480785) < 1.0, str(qa)
            assert numpy.abs(qa.data['min'] + 0.7681701460717593) < 1.0, str(qa)
            assert numpy.abs(r[1]-415950.0) < 1e-7, str(qa)
            
if __name__ == '__main__':
    unittest.main()
