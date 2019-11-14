""" Unit tests for pipelines expressed via dask.delayed


"""
import functools
import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from processing_components.image.operations import export_image_to_fits, smooth_image
from processing_components.imaging.base import predict_2d, invert_2d, predict_skycomponent_visibility
from processing_components.simulation.testing_support import ingest_unittest_visibility, \
    create_unittest_model, create_unittest_components
from processing_components.simulation.configurations import create_named_configuration
from processing_components.skycomponent.operations import find_skycomponents, find_nearest_skycomponent, \
    insert_skycomponent
from processing_components.imaging.primary_beams import create_pb_generic
from processing_components.imaging.weighting import taper_visibility_gaussian, taper_visibility_tukey, \
    weight_visibility
from processing_components.griddata.kernels import create_awterm_convolutionfunction

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImaging(unittest.TestCase):
    def setUp(self):
        
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.persist = False
    
    def actualSetUp(self, freqwin=1, block=False, dospectral=True, dopol=False, zerow=False):
        
        self.npixel = 512
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = freqwin
        self.vis = list()
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        
        if freqwin > 1:
            self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
            self.channelwidth = numpy.array(freqwin * [self.frequency[1] - self.frequency[0]])
        else:
            self.frequency = numpy.array([1e8])
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
        self.vis = ingest_unittest_visibility(self.low,
                                              [self.frequency],
                                              [self.channelwidth],
                                              self.times,
                                              self.vis_pol,
                                              self.phasecentre, block=block,
                                              zerow=zerow)
        
        self.model = create_unittest_model(self.vis, self.image_pol, npixel=self.npixel)
        
        self.components = create_unittest_components(self.model, flux)
        
        self.model = insert_skycomponent(self.model, self.components)
        
        self.vis = predict_skycomponent_visibility(self.vis, self.components)
        
        # Calculate the model convolved with a Gaussian.
        
        self.cmodel = smooth_image(self.model)
        if self.persist: export_image_to_fits(self.model, '%s/test_imaging_model.fits' % self.dir)
        if self.persist: export_image_to_fits(self.cmodel, '%s/test_imaging_cmodel.fits' % self.dir)
    
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
    
    def _predict_base(self, fluxthreshold=1.0, gcf=None, cf=None, name='predict_2d', gcfcf=None, **kwargs):
        
        vis = predict_2d(self.vis, self.model, gcfcf = gcfcf, **kwargs)
        vis.data['vis'] = self.vis.data['vis'] - vis.data['vis']
        dirty = invert_2d(vis, self.model, dopsf=False, normalize=True, gcfcf = gcfcf)
        
        if self.persist: export_image_to_fits(dirty[0], '%s/test_imaging_%s_residual.fits' %
                             (self.dir, name))
        assert numpy.max(numpy.abs(dirty[0].data)), "Residual image is empty"

        maxabs = numpy.max(numpy.abs(dirty[0].data))
        assert maxabs < fluxthreshold, "Error %.3f greater than fluxthreshold %.3f " % (maxabs, fluxthreshold)
    
    def _invert_base(self, fluxthreshold=1.0, positionthreshold=1.0, check_components=True,
                     name='predict_2d', gcfcf=None, **kwargs):
        
        dirty = invert_2d(self.vis, self.model, dopsf=False, normalize=True, gcfcf = gcfcf, **kwargs)
        
        if self.persist: export_image_to_fits(dirty[0], '%s/test_imaging_%s_dirty.fits' %
                             (self.dir, name))
        
        assert numpy.max(numpy.abs(dirty[0].data)), "Image is empty"
        
        if check_components:
            self._checkcomponents(dirty[0], fluxthreshold, positionthreshold)

    def test_predict_2d(self):
        self.actualSetUp(zerow=True)
        self._predict_base(name='predict_2d')

    def test_invert_2d(self):
        self.actualSetUp(zerow=True)
        self._invert_base(name='invert_2d', positionthreshold=2.0, check_components=True)

    def test_predict_awterm(self):
        self.actualSetUp(zerow=False)
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        gcfcf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=100, wstep=8.0,
                                                    oversampling=4, support=100, use_aaf=True)
        self._predict_base(name='predict_awterm', fluxthreshold=35.0, gcfcf = gcfcf)

    def test_invert_awterm(self):
        self.actualSetUp(zerow=False)
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        gcfcf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=100, wstep=8.0,
                                                    oversampling=4, support=100, use_aaf=True)
        self._invert_base(name='invert_awterm', positionthreshold=35.0, check_components=False, gcfcf = gcfcf)

    def test_predict_wterm(self):
        self.actualSetUp(zerow=False)
        gcfcf = create_awterm_convolutionfunction(self.model, nw=100, wstep=8.0,
                                                    oversampling=8, support=100, use_aaf=True)
        self._predict_base(name='predict_wterm', gcfcf = gcfcf, fluxthreshold=5.0)

    def test_invert_wterm(self):
        self.actualSetUp(zerow=False)
        gcfcf = create_awterm_convolutionfunction(self.model, nw=100, wstep=8.0,
                                                    oversampling=8, support=100, use_aaf=True)
        self._invert_base(name='invert_wterm', positionthreshold=35.0, check_components=False, gcfcf = gcfcf)


if __name__ == '__main__':
    unittest.main()
