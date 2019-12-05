""" Unit tests for pipelines expressed via dask.delayed


"""
import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from processing_components.image.operations import export_image_to_fits, smooth_image
from processing_components.imaging.base import predict_skycomponent_visibility
from processing_components.simulation.configurations import create_named_configuration
from processing_components.simulation.testing_support import ingest_unittest_visibility, \
    create_unittest_model, create_unittest_components
from processing_components.skycomponent.operations import find_skycomponents, find_nearest_skycomponent, \
    insert_skycomponent
from processing_components.visibility.coalesce import convert_blockvisibility_to_visibility
from processing_components.visibility.base import copy_visibility

try:
    import nifty_gridder
    run_ng_tests = True
#            except ModuleNotFoundError:
except ImportError:
    run_ng_tests = False

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImagingNG(unittest.TestCase):
    def setUp(self):
        
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.persist = True
    
    def actualSetUp(self, freqwin=1, block=True, dospectral=True, dopol=False, zerow=False, do_shift=False):
        
        self.npixel = 512
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = freqwin
        self.blockvis = list()
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        
        if freqwin > 1:
            self.frequency = numpy.linspace(0.99e8, 1.01e8, self.freqwin)
            self.channelwidth = numpy.array(freqwin * [self.frequency[1] - self.frequency[0]])
        else:
            self.frequency = numpy.array([1e8])
            self.channelwidth = numpy.array([1e6])
        
        if dopol:
            self.blockvis_pol = PolarisationFrame('linear')
            self.image_pol = PolarisationFrame('stokesIQUV')
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        else:
            self.blockvis_pol = PolarisationFrame('stokesI')
            self.image_pol = PolarisationFrame('stokesI')
            f = numpy.array([100.0])
        
        if dospectral:
            flux = numpy.array([f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency])
        else:
            flux = numpy.array([f])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
        self.blockvis = ingest_unittest_visibility(self.low,
                                                   self.frequency,
                                                   self.channelwidth,
                                                   self.times,
                                                   self.blockvis_pol,
                                                   self.phasecentre,
                                                   block=block,
                                                   zerow=zerow)
        
        self.vis = convert_blockvisibility_to_visibility(self.blockvis)
        
        self.model = create_unittest_model(self.vis, self.image_pol, npixel=self.npixel, nchan=freqwin)
        
        self.components = create_unittest_components(self.model, flux)
        
        self.model = insert_skycomponent(self.model, self.components)
        
        self.blockvis = predict_skycomponent_visibility(self.blockvis, self.components)
        
        # Calculate the model convolved with a Gaussian.
        
        self.cmodel = smooth_image(self.model)
        if self.persist: export_image_to_fits(self.model, '%s/test_imaging_ng_model.fits' % self.dir)
        if self.persist: export_image_to_fits(self.cmodel, '%s/test_imaging_ng_cmodel.fits' % self.dir)
    
    def _checkcomponents(self, dirty, fluxthreshold=0.6, positionthreshold=0.1):
        comps = find_skycomponents(dirty, fwhm=1.0, threshold=10 * fluxthreshold, npixels=5)
        assert len(comps) == len(self.components), "Different number of components found: original %d, recovered %d" % \
                                                   (len(self.components), len(comps))
        cellsize = abs(dirty.wcs.wcs.cdelt[0])
        
        for comp in comps:
            # Check for agreement in direction
            ocomp, separation = find_nearest_skycomponent(comp.direction, self.components)
            assert separation / cellsize < positionthreshold, "Component differs in position %.3f pixels" % \
                                                              separation / cellsize
    
    def _predict_base(self, fluxthreshold=1.0, name='predict_ng', **kwargs):
        
        from processing_components.imaging.ng import predict_ng, invert_ng
        original_vis = copy_visibility(self.blockvis)
        vis = predict_ng(self.blockvis, self.model, **kwargs)
        vis.data['vis'] = vis.data['vis'] - original_vis.data['vis']
        dirty = invert_ng(vis, self.model, dopsf=False, normalize=True, **kwargs)
        
        import matplotlib.pyplot as plt
        from processing_components.image.operations import show_image
        npol = dirty[0].shape[1]
        for pol in range(npol):
            plt.clf()
            show_image(dirty[0], pol=pol)
            plt.show(block=False)

        if self.persist: export_image_to_fits(dirty[0], '%s/test_imaging_ng_%s_residual.fits' %
                                              (self.dir, name))

        # assert numpy.max(numpy.abs(dirty[0].data)), "Residual image is empty"
        
        maxabs = numpy.max(numpy.abs(dirty[0].data))
        assert maxabs < fluxthreshold, "Error %.3f greater than fluxthreshold %.3f " % (maxabs, fluxthreshold)
    
    def _invert_base(self, fluxthreshold=1.0, positionthreshold=1.0, check_components=True,
                     name='predict_ng', **kwargs):
        
        # dirty = invert_ng(self.blockvis, self.model, dopsf=False, normalize=True, **kwargs)
        from processing_components.imaging.ng import predict_ng, invert_ng
        dirty = invert_ng(self.blockvis, self.model, normalize=True, verbosity=2, **kwargs)

        if self.persist: export_image_to_fits(dirty[0], '%s/test_imaging_ng_%s_dirty.fits' %
                                              (self.dir, name))
        
        import matplotlib.pyplot as plt
        from processing_components.image.operations import show_image
        npol = dirty[0].shape[1]
        for pol in range(npol):
            plt.clf()
            show_image(dirty[0], pol=pol)
            plt.show(block=False)


        assert numpy.max(numpy.abs(dirty[0].data)), "Image is empty"

        if check_components:
            self._checkcomponents(dirty[0], fluxthreshold, positionthreshold)

    @unittest.skipUnless(run_ng_tests, "requires the nifty_gridder module")
    def test_predict_ng(self):
        self.actualSetUp()
        self._predict_base(name='predict')

    @unittest.skipUnless(run_ng_tests, "requires the nifty_gridder module")
    def test_invert_ng(self):
        self.actualSetUp()
        self._invert_base(name='invert', positionthreshold=2.0, check_components=True)

    @unittest.skipUnless(run_ng_tests, "requires the nifty_gridder module")
    def test_predict_ng_pol(self):
        self.actualSetUp(dopol=True)
        self._predict_base(name='predict_pol')

    @unittest.skipUnless(run_ng_tests, "requires the nifty_gridder module")
    def test_invert_ng_pol(self):
        self.actualSetUp(dopol=True)
        self._invert_base(name='invert_pol', positionthreshold=2.0, check_components=False)

    @unittest.skipUnless(run_ng_tests, "requires the nifty_gridder module")
    def test_predict_ng_spec(self):
        self.actualSetUp(dospectral=True, freqwin=5)
        self._predict_base(name='predict_spec')

    @unittest.skipUnless(run_ng_tests, "requires the nifty_gridder module")
    def test_invert_ng_spec(self):
        self.actualSetUp(dospectral=True, freqwin=5)
        self._invert_base(name='invert_spec', positionthreshold=2.0, check_components=False)

    @unittest.skipUnless(run_ng_tests, "requires the nifty_gridder module")
    def test_predict_ng_spec_pol(self):
        self.actualSetUp(dospectral=True, freqwin=5, dopol=True)
        self._predict_base(name='predict_spec_pol')

    @unittest.skipUnless(run_ng_tests, "requires the nifty_gridder module")
    def test_invert_ng_spec_pol(self):
        self.actualSetUp(dospectral=True, freqwin=5, dopol=True)
        self._invert_base(name='invert_spec_pol', positionthreshold=2.0, check_components=False)


if __name__ == '__main__':
    unittest.main()
