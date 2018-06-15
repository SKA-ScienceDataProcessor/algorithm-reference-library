""" Unit tests for data model helpers. The helpers facilitate persistence of data models
using HDF5


"""

import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.data_model_helpers import import_visibility_from_hdf5, export_visibility_to_hdf5, \
    import_blockvisibility_from_hdf5, export_blockvisibility_to_hdf5, \
    import_gaintable_from_hdf5, export_gaintable_to_hdf5, \
    import_image_from_hdf5, export_image_to_hdf5, \
    import_skycomponent_from_hdf5, export_skycomponent_to_hdf5, \
    import_skymodel_from_hdf5, export_skymodel_to_hdf5
from data_models.memory_data_models import Skycomponent, SkyModel
from data_models.polarisation import PolarisationFrame
from processing_components.calibration.operations import create_gaintable_from_blockvisibility
from processing_components.imaging.base import predict_skycomponent_visibility
from processing_components.simulation.testing_support import create_named_configuration, \
    simulate_gaintable, create_test_image
from processing_components.visibility.base import create_visibility, create_blockvisibility


class TestDataModelHelpers(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 100.0)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, 3)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.comp = Skycomponent(direction=self.compabsdirection, frequency=self.frequency, flux=self.flux)
    
    def test_readwritevisibility(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre,
                                     polarisation_frame=PolarisationFrame("linear"),
                                     weight=1.0)
        self.vis = predict_skycomponent_visibility(self.vis, self.comp)
        export_visibility_to_hdf5(self.vis, '%s/test_visibility.hdf' % self.dir)
        newvis = import_visibility_from_hdf5('%s/test_visibility.hdf' % self.dir)
        
        assert str(newvis) == str(self.vis), "Original %s, import %s" % (str(newvis), str(self.vis))
        assert numpy.array_equal(newvis.frequency, self.vis.frequency)
        assert newvis.data.shape == self.vis.data.shape
        assert numpy.array_equal(newvis.frequency, self.vis.frequency)
        assert numpy.max(numpy.abs(self.vis.vis - newvis.vis)) < 1e-15
        assert numpy.max(numpy.abs(self.vis.uvw - newvis.uvw)) < 1e-15
        assert numpy.abs(newvis.configuration.location.x.value - self.vis.configuration.location.x.value) < 1e-15
        assert numpy.abs(newvis.configuration.location.y.value - self.vis.configuration.location.y.value) < 1e-15
        assert numpy.abs(newvis.configuration.location.z.value - self.vis.configuration.location.z.value) < 1e-15
        assert numpy.max(numpy.abs(newvis.configuration.xyz - self.vis.configuration.xyz)) < 1e-15
    
    def test_readwriteblockvisibility(self):
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre,
                                          polarisation_frame=PolarisationFrame("linear"),
                                          weight=1.0)
        self.vis = predict_skycomponent_visibility(self.vis, self.comp)
        export_blockvisibility_to_hdf5(self.vis, '%s/test_blockvisibility.hdf' % self.dir)
        newvis = import_blockvisibility_from_hdf5('%s/test_blockvisibility.hdf' % self.dir)
        
        assert numpy.array_equal(newvis.frequency, self.vis.frequency)
        assert newvis.data.shape == self.vis.data.shape
        assert numpy.max(numpy.abs(self.vis.vis - newvis.vis)) < 1e-15
        assert numpy.max(numpy.abs(self.vis.uvw - newvis.uvw)) < 1e-15
        assert numpy.abs(newvis.configuration.location.x.value - self.vis.configuration.location.x.value) < 1e-15
        assert numpy.abs(newvis.configuration.location.y.value - self.vis.configuration.location.y.value) < 1e-15
        assert numpy.abs(newvis.configuration.location.z.value - self.vis.configuration.location.z.value) < 1e-15
        assert numpy.max(numpy.abs(newvis.configuration.xyz - self.vis.configuration.xyz)) < 1e-15
    
    def test_readwritegaintable(self):
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre,
                                          polarisation_frame=PolarisationFrame("linear"),
                                          weight=1.0)
        gt = create_gaintable_from_blockvisibility(self.vis, timeslice='auto')
        gt = simulate_gaintable(gt, phase_error=1.0, amplitude_error=0.1)
        export_gaintable_to_hdf5(gt, '%s/test_gaintable.hdf' % self.dir)
        newgt = import_gaintable_from_hdf5('%s/test_gaintable.hdf' % self.dir)
        
        assert gt.data.shape == newgt.data.shape
        assert numpy.max(numpy.abs(gt.gain - newgt.gain)) < 1e-15
    
    def test_readwriteimage(self):
        im = create_test_image()
        export_image_to_hdf5(im, '%s/test_image.hdf' % self.dir)
        newim = import_image_from_hdf5('%s/test_image.hdf' % self.dir)
        
        assert newim.data.shape == im.data.shape
        assert numpy.max(numpy.abs(im.data - newim.data)) < 1e-15

    def test_readwriteskycomponent(self):
        export_skycomponent_to_hdf5(self.comp, '%s/test_skycomponent.hdf' % self.dir)
        newsc = import_skycomponent_from_hdf5('%s/test_skycomponent.hdf' % self.dir)
    
        assert newsc.flux.shape == self.comp.flux.shape
        assert numpy.max(numpy.abs(newsc.flux - self.comp.flux)) < 1e-15

    def test_readwriteskymodel(self):
        im = create_test_image()
        sm = SkyModel(components=[self.comp], images=[im, im])
        export_skymodel_to_hdf5(sm, '%s/test_skymodel.hdf' % self.dir)
        newsm = import_skymodel_from_hdf5('%s/test_skymodel.hdf' % self.dir)
    
        assert newsm.components[0].flux.shape == self.comp.flux.shape
        assert newsm.images[0].data.shape == im.data.shape
        assert numpy.max(numpy.abs(newsm.images[0].data - im.data)) < 1e-15


if __name__ == '__main__':
    unittest.main()
