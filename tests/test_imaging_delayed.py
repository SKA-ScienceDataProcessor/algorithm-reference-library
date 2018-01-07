""" Unit tests for pipelines expressed via dask.delayed


"""

import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord
from dask import delayed

from arl.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility
from arl.data.polarisation import PolarisationFrame
from arl.graphs.delayed import create_zero_vis_graph_list, create_subtract_vis_graph_list, \
    create_predict_graph, create_invert_graph
from arl.image.operations import qa_image, export_image_to_fits, copy_image, create_empty_image_like
from arl.imaging import create_image_from_visibility, predict_skycomponent_blockvisibility, \
    predict_skycomponent_visibility
from arl.imaging.weighting import weight_visibility
from arl.skycomponent.operations import create_skycomponent, insert_skycomponent
from arl.util.testing_support import create_named_configuration
from arl.util.testing_support import simulate_gaintable
from arl.visibility.base import create_visibility, create_blockvisibility
from arl.visibility.operations import qa_visibility


class TestDaskGraphs(unittest.TestCase):
    def setUp(self):
        
        self.compute = True
        
        self.results_dir = './test_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.params = {'npixel': 512,
                       'nchan': 1,
                       'reffrequency': 1e8,
                       'facets': 1,
                       'padding': 2,
                       'oversampling': 2,
                       'kernel': '2d',
                       'wstep':4.0,
                       'wstack':4.0}

    
    def actualSetUp(self, add_errors=False, freqwin=7, block=False, dospectral=True, dopol=False):
        self.freqwin = freqwin
        self.vis_graph_list = list()
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        
        for freq in self.frequency:
            self.vis_graph_list.append(delayed(self.ingest_visibility)(freq, times=self.times,
                                                                  add_errors=add_errors,
                                                                  block=block))
        
        self.nvis = len(self.vis_graph_list)
        self.model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2])
        
        self.wstep = 10.0
        self.vis_slices = 2 * int(numpy.ceil(numpy.max(numpy.abs(self.vis_graph_list[0].compute().w)) / self.wstep)) + 1

    def ingest_visibility(self, freq=1e8, chan_width=1e6, times=None, reffrequency=None,
                          add_errors=False, block=False):
        if times is None:
            times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 5)
        
        if reffrequency is None:
            reffrequency = [1e8]
        lowcore = create_named_configuration('LOWBD2', rmax=1000.0)
        frequency = numpy.array([freq])
        channel_bandwidth = numpy.array([chan_width])
        
        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        if block:
            vt = create_blockvisibility(lowcore, times, frequency, channel_bandwidth=channel_bandwidth,
                                        weight=1.0, phasecentre=phasecentre,
                                        polarisation_frame=PolarisationFrame("stokesI"))
        else:
            vt = create_visibility(lowcore, times, frequency, channel_bandwidth=channel_bandwidth,
                                   weight=1.0, phasecentre=phasecentre,
                                   polarisation_frame=PolarisationFrame("stokesI"))
        cellsize = 0.001
        model = create_image_from_visibility(vt, npixel=self.params['npixel'], cellsize=cellsize, npol=1,
                                             frequency=reffrequency, phasecentre=phasecentre,
                                             polarisation_frame=PolarisationFrame("stokesI"))
        flux = numpy.array([[100.0]])
        facets = 4
        
        rpix = model.wcs.wcs.crpix - 1.0
        spacing_pixels = self.params['npixel'] // facets
        centers = [-1.5, -0.5, 0.5, 1.5]
        comps = list()
        for iy in centers:
            for ix in centers:
                p = int(round(rpix[0] + ix * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[0]))), \
                    int(round(rpix[1] + iy * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[1])))
                sc = pixel_to_skycoord(p[0], p[1], model.wcs, origin=1)
                comp = create_skycomponent(flux=flux, frequency=frequency, direction=sc,
                                           polarisation_frame=PolarisationFrame("stokesI"))
                comps.append(comp)
        if block:
            predict_skycomponent_blockvisibility(vt, comps)
        else:
            predict_skycomponent_visibility(vt, comps)
        insert_skycomponent(model, comps)
        self.model = copy_image(model)
        self.empty_model = create_empty_image_like(model)
        
        export_image_to_fits(model, '%s/test_graphs_delayed_model.fits' % (self.results_dir))
        
        if add_errors:
            # These will be the same for all calls
            numpy.random.seed(180555)
            gt = create_gaintable_from_blockvisibility(vt)
            gt = simulate_gaintable(gt, phase_error=1.0, amplitude_error=0.0)
            vt = apply_gaintable(vt, gt)
        return vt
    
    def get_LSM(self, vt, cellsize=0.001, reffrequency=None, flux=0.0):
        if reffrequency is None:
            reffrequency = [1e8]
        model = create_image_from_visibility(vt, npixel=self.params['npixel'], cellsize=cellsize, npol=1,
                                             frequency=reffrequency,
                                             polarisation_frame=PolarisationFrame("stokesI"))
        model.data[..., 31, 31] = flux
        return model
    
    def _predict_base(self, context, extra='', fluxthreshold=5.0):
        flux_model_graph = delayed(self.get_LSM)(self.vis_graph_list[self.nvis // 2], flux=100.0)
        zero_vis_graph_list = create_zero_vis_graph_list(self.vis_graph_list)
        predicted_vis_graph_list = \
            create_predict_graph(self.vis_graph_list, flux_model_graph, context=context, **self.params)
        if self.compute:
            qa = qa_visibility(predicted_vis_graph_list[0].compute())
            print(qa)
    
    def _invert_base(self, context='2d', extra='', positionthreshold=1.0, check_components=True):
        dirty_graph = create_invert_graph(self.vis_graph_list, self.model_graph,
                                          context=context,
                                          dopsf=False, normalize=True,
                                          **self.params)
        
        if self.compute:
            dirty = dirty_graph[0].compute()
            export_image_to_fits(dirty[0], '%s/test_imaging_graph_invert_%s%s_dirty.fits' %
                                 (self.results_dir, context, extra,))
            qa = qa_image(dirty[0])
            
            assert numpy.abs(qa.data['max'] - 100.0) < 5.0, str(qa)
            assert numpy.abs(qa.data['min'] + 2.0) < 5.0, str(qa)
    
    def test_predict_2d(self):
        self.actualSetUp()
        self._predict_base(context='2d', fluxthreshold=5.0)

    @unittest.skip("Intrinsically unstable")
    def test_predict_facets_wstack(self):
        self.params['wstack'] = 4.0
        self.params['facets'] = 21
        self.params['npixel'] = 21 * 32
        self.params['padding'] = 4
        self.actualSetUp()
        self._predict_base(context='facets_wstack', fluxthreshold=5.0)
    
    #@unittest.skip("Intrinsically unstable")
    def test_predict_timeslice(self):
        self.params['timeslice'] = 'auto'
        self.actualSetUp()
        self._predict_base(context='timeslice', fluxthreshold=5.0)
    
    @unittest.skip("Intrinsically unstable")
    def test_predict_timeslice_wprojection(self):
        self.actualSetUp()
        self.params['kernel'] = 'wprojection'
        self.params['wstep'] = 4.0
        self.params['timeslice'] = 1e5
        self._predict_base(context='timeslice', extra='_wprojection', fluxthreshold=5.0)
    
    def test_predict_wprojection(self):
        self.actualSetUp()
        self.params['wstep'] = 4.0
        self._predict_base(context='2d', extra='_wprojection', fluxthreshold=5.0)
    
    def test_predict_wstack(self):
        self.actualSetUp()
        self.params['wstack'] = 4.0
        self._predict_base(context='wstack', fluxthreshold=5.0)
    
    def test_predict_wstack_wprojection(self):
        self.actualSetUp()
        self.params['kernel'] = 'wprojection'
        self.params['wstack'] = 5 * 4.0
        self.params['wstep'] = 4.0
        self._predict_base(context='wstack', fluxthreshold=5.0)
    
    def test_predict_wstack_spectral(self):
        self.params['wstack'] = 4.0
        self.actualSetUp(dospectral=True)
        self._predict_base(context='wstack', extra='_spectral', fluxthreshold=7.0)
    
    def test_predict_wstack_spectral_pol(self):
        self.params['wstack'] = 4.0
        self.actualSetUp(dospectral=True, dopol=True)
        self._predict_base(context='wstack', extra='_spectral', fluxthreshold=7.0)

    def test_invert_2d(self):
        self.actualSetUp()
        self._invert_base(context='2d', positionthreshold=1.0, check_components=False)

    def test_invert_facets(self):
        self.params['facets'] = 21
        self.params['npixel'] = 21 * 32
        self.params['padding'] = 4
        self.actualSetUp()
        self._invert_base(context='facets', positionthreshold=1.0, check_components=True)

    def test_invert_facets_timeslice(self):
        # Gaps in images lead to missing sources: use extra padding?
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['padding'] = 4
        self.params['timeslice'] = 1e5
        self.params['remove_shift'] = True
        self.actualSetUp()
        self._invert_base(context='facets_timeslice', check_components=True,
                          positionthreshold=2.0)
    
    @unittest.skip("Seems to be correcting twice!")
    def test_invert_facets_wprojection(self):
        self.params['facets'] = 9
        self.params['npixel'] = 64 * 9
        self.params['wstep'] = 4.0
        self.params['remove_shift'] = True
        self.actualSetUp()
        self._invert_base(context='facets', extra='_wprojection', check_components=True,
                          positionthreshold=1.0)
    
    def test_invert_facets_wstack(self):
        self.params['wstack'] = 4.0
        self.params['npixel'] = 9 * 64
        self.params['facets'] = 9
        self.params['remove_shift'] = True
        self.actualSetUp()
        self._invert_base(context='facets_wstack', positionthreshold=1.0,
                          check_components=False)
    
    def test_invert_timeslice(self):
        self.actualSetUp()
        self._invert_base(context='timeslice', positionthreshold=8.0, check_components=True)
    
    def test_invert_timeslice_wprojection(self):
        self.actualSetUp()
        self.params['wstep'] = 4.0
        self._invert_base(context='timeslice', extra='_wprojection', positionthreshold=1.0,
                          check_components=True)
    
    def test_invert_wprojection(self):
        self.actualSetUp()
        self.params['kernel'] = 'wprojection'
        self.params['wstep'] = 4.0
        self._invert_base(context='2d', extra='wprojection', positionthreshold=1.0)
    
    def test_invert_wprojection_wstack(self):
        self.actualSetUp()
        self.params['kernel'] = 'wprojection'
        self.params['wstack'] = 5 * 4.0
        self.params['wstep'] = 4.0
        self._invert_base(context='wstack', extra='wprojection', positionthreshold=1.0)
    
    def test_invert_wstack(self):
        self.actualSetUp()
        self.params['wstack'] = 4.0
        self._invert_base(context='wstack', positionthreshold=1.0)
    
    def test_invert_wstack_spectral(self):
        self.actualSetUp(dospectral=True)
        self.params['wstack'] = 4.0
        self._invert_base(context='wstack', extra='_spectral', positionthreshold=1.0)
    
    def test_invert_wstack_spectral_pol(self):
        self.actualSetUp(dospectral=True, dopol=True)
        self.params['wstack'] = 4.0
        self._invert_base(context='wstack', extra='_spectral_pol', positionthreshold=1.0)
    
if __name__ == '__main__':
    unittest.main()
