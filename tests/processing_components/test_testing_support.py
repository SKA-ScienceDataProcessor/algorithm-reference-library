"""Unit tests for testing support


"""

import logging
import sys
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.memory_data_models import Skycomponent
from data_models.polarisation import PolarisationFrame
from processing_components.image.operations import export_image_to_fits
from processing_components.imaging.base import predict_skycomponent_visibility
from processing_components.imaging.primary_beams import create_low_test_beam
from processing_components.simulation.testing_support import create_test_image_from_s3, create_test_image, create_blockvisibility_iterator, create_low_test_image_from_gleam, \
    create_low_test_skycomponents_from_gleam, create_low_test_skymodel_from_gleam, create_test_skycomponents_from_s3
from processing_components.simulation.configurations import create_named_configuration
from processing_components.visibility.base import create_visibility, create_blockvisibility
from processing_components.visibility.coalesce import convert_blockvisibility_to_visibility
from processing_components.visibility.operations import append_visibility

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestTesting_Support(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.frequency = numpy.linspace(0.8e8, 1.2e8, 5)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
        self.flux = numpy.array([[100.0], [100.0], [100.0], [100.0], [100.0]])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.config = create_named_configuration('LOWBD2-CORE')
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        nants = self.config.xyz.shape[0]
        assert nants > 1
        assert len(self.config.names) == nants
        assert len(self.config.mount) == nants
    
    def createVis(self, config, dec=-35.0, rmax=None):
        self.config = create_named_configuration(config, rmax=rmax)
        self.phasecentre = SkyCoord(ra=+15 * u.deg, dec=dec * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_visibility(self.config, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('stokesI'))
    
    def test_create_test_image(self):
        im = create_test_image(canonical=False)
        assert len(im.data.shape) == 2
        im = create_test_image(canonical=True)
        assert len(im.data.shape) == 4
        im = create_test_image(canonical=True, frequency=numpy.array([1e8]),
                               polarisation_frame=PolarisationFrame(
                                   'stokesI'))
        assert len(im.data.shape) == 4
        assert im.data.shape[0] == 1
        assert im.data.shape[1] == 1
        im = create_test_image(canonical=True, frequency=numpy.array([1e8]),
                               polarisation_frame=PolarisationFrame(
                                   'stokesIQUV'))
        assert len(im.data.shape) == 4
        assert im.data.shape[0] == 1
        assert im.data.shape[1] == 4
    
    def test_create_low_test_skymodel_from_gleam(self):
        sm = create_low_test_skymodel_from_gleam(npixel=256, cellsize=0.001, frequency=self.frequency,
                                                 channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                                 kind='cubic', flux_limit=0.3, flux_threshold=1.0)
        
        im = sm.image
        assert im.data.shape[0] == 5
        assert im.data.shape[1] == 1
        assert im.data.shape[2] == 256
        assert im.data.shape[3] == 256
        export_image_to_fits(im, '%s/test_test_support_low_gleam.fits' % (self.dir))
        
        comp = sm.components
        assert len(comp) == 79, len(comp)
        assert comp[0].name == 'GLEAM J004616-420739', comp[0].name
        assert comp[-1].name == 'GLEAM J011535-314620', comp[-1].name
    
    def test_create_low_test_image_from_gleam(self):
        im = create_low_test_image_from_gleam(npixel=256, cellsize=0.001,
                                              channel_bandwidth=self.channel_bandwidth,
                                              frequency=self.frequency,
                                              phasecentre=self.phasecentre,
                                              kind='cubic', flux_limit=0.3)
        assert im.data.shape[0] == 5
        assert im.data.shape[1] == 1
        assert im.data.shape[2] == 256
        assert im.data.shape[3] == 256
        export_image_to_fits(im, '%s/test_test_support_low_gleam.fits' % (self.dir))
    
    def test_create_low_test_image_from_gleam_with_pb(self):
        im = create_low_test_image_from_gleam(npixel=256, cellsize=0.001,
                                              channel_bandwidth=self.channel_bandwidth,
                                              frequency=self.frequency,
                                              phasecentre=self.phasecentre,
                                              kind='cubic',
                                              applybeam=True, flux_limit=1.0)
        assert im.data.shape[0] == 5
        assert im.data.shape[1] == 1
        assert im.data.shape[2] == 256
        assert im.data.shape[3] == 256
        export_image_to_fits(im, '%s/test_test_support_low_gleam_with_pb.fits' % (self.dir))
    
    def test_create_low_test_skycomponents_from_gleam(self):
        sc = create_low_test_skycomponents_from_gleam(flux_limit=1.0,
                                                      phasecentre=SkyCoord("17h20m31s", "-00d58m45s"),
                                                      polarisation_frame=PolarisationFrame("stokesI"),
                                                      frequency=self.frequency, kind='cubic', radius=0.001)
        assert len(sc) == 1, "Only expected one source, actually found %d" % len(sc)
        assert sc[0].name == 'GLEAM J172031-005845'
        self.assertAlmostEqual(sc[0].flux[0, 0], 357.2599499089219, 7)
    
    def test_create_test_skycomponents_from_s3(self):
        self.frequency = numpy.linspace(0.8e9, 1.2e9, 5)
        sc = create_test_skycomponents_from_s3(flux_limit=3.0,
                                               phasecentre=self.phasecentre,
                                               polarisation_frame=PolarisationFrame("stokesI"),
                                               frequency=self.frequency, radius=0.1)
        assert len(sc) == 9, "Only expected nine sources, actually found %d" % len(sc)
        assert sc[0].name == 'S3_36133023'
        self.assertAlmostEqual(sc[0].flux[0, 0], 4.54336938, 7)
    
    def test_create_test_image_from_s3_low(self):
        im = create_test_image_from_s3(npixel=1024, channel_bandwidth=numpy.array([1e6]),
                                       frequency=numpy.array([1e8]),
                                       phasecentre=self.phasecentre, fov=10)
        assert im.data.shape[0] == 1
        assert im.data.shape[1] == 1
        assert im.data.shape[2] == 1024
        assert im.data.shape[3] == 1024
        export_image_to_fits(im, '%s/test_test_support_low_s3.fits' % (self.dir))
    
    def test_create_test_image_from_s3_mid(self):
        im = create_test_image_from_s3(npixel=1024, channel_bandwidth=numpy.array([1e6]),
                                       frequency=numpy.array([1e9]),
                                       phasecentre=self.phasecentre,
                                       flux_limit=2e-3)
        assert im.data.shape[0] == 1
        assert im.data.shape[1] == 1
        assert im.data.shape[2] == 1024
        assert im.data.shape[3] == 1024
        export_image_to_fits(im, '%s/test_test_support_mid_s3.fits' % (self.dir))
    
    def test_create_test_image_s3_spectral(self):
        im = create_test_image_from_s3(npixel=1024, channel_bandwidth=numpy.array([1e6, 1e6, 1e6]),
                                       frequency=numpy.array([1e8 - 1e6, 1e8, 1e8 + 1e6]),
                                       phasecentre=self.phasecentre, fov=10,
                                       flux_limit=2e-3)
        assert im.data.shape[0] == 3
        assert im.data.shape[1] == 1
        assert im.data.shape[2] == 1024
        assert im.data.shape[3] == 1024
    
    def test_create_low_test_image_s3_spectral_polarisation(self):
        
        im = create_test_image_from_s3(npixel=1024, channel_bandwidth=numpy.array([1e6, 1e6, 1e6]),
                                       polarisation_frame=PolarisationFrame("stokesIQUV"),
                                       frequency=numpy.array([1e8 - 1e6, 1e8, 1e8 + 1e6]), fov=10)
        assert im.data.shape[0] == 3
        assert im.data.shape[1] == 4
        assert im.data.shape[2] == 1024
        assert im.data.shape[3] == 1024
        export_image_to_fits(im, '%s/test_test_support_low_s3.fits' % (self.dir))
    
    def test_create_low_test_beam(self):
        im = create_test_image(canonical=True, cellsize=0.002,
                               frequency=numpy.array([1e8 - 5e7, 1e8, 1e8 + 5e7]),
                               channel_bandwidth=numpy.array([5e7, 5e7, 5e7]),
                               polarisation_frame=PolarisationFrame("stokesIQUV"),
                               phasecentre=self.phasecentre)
        bm = create_low_test_beam(model=im)
        export_image_to_fits(bm, '%s/test_test_support_low_beam.fits' % (self.dir))
        
        assert bm.data.shape[0] == 3
        assert bm.data.shape[1] == 4
        assert bm.data.shape[2] == im.data.shape[2]
        assert bm.data.shape[3] == im.data.shape[3]
        # Check to see if the beam scales as expected
        for i in [30, 40]:
            assert numpy.max(numpy.abs(bm.data[0, 0, 128, 128 - 2 * i] - bm.data[1, 0, 128, 128 - i])) < 0.02
            assert numpy.max(numpy.abs(bm.data[0, 0, 128, 128 - 3 * i] - bm.data[2, 0, 128, 128 - i])) < 0.02
            assert numpy.max(numpy.abs(bm.data[0, 0, 128 - 2 * i, 128] - bm.data[1, 0, 128 - i, 128])) < 0.02
            assert numpy.max(numpy.abs(bm.data[0, 0, 128 - 3 * i, 128] - bm.data[2, 0, 128 - i, 128])) < 0.02
    
    def test_create_vis_iter(self):
        vis_iter = create_blockvisibility_iterator(self.config, self.times, self.frequency,
                                                   channel_bandwidth=self.channel_bandwidth,
                                                   phasecentre=self.phasecentre,
                                                   weight=1.0, polarisation_frame=PolarisationFrame('stokesI'),
                                                   integration_time=30.0, number_integrations=3)
        
        fullvis = None
        totalnvis = 0
        for i, vis in enumerate(vis_iter):
            assert vis.nvis
            if i == 0:
                fullvis = vis
                totalnvis = vis.nvis
            else:
                fullvis = append_visibility(fullvis, vis)
                totalnvis += vis.nvis
        
        assert fullvis.nvis == totalnvis
    
    def test_create_vis_iter_with_model(self):
        model = create_test_image(canonical=True, cellsize=0.001, frequency=self.frequency,
                                  phasecentre=self.phasecentre)
        comp = Skycomponent(direction=self.phasecentre, frequency=self.frequency, flux=self.flux,
                            polarisation_frame=PolarisationFrame('stokesI'))
        vis_iter = create_blockvisibility_iterator(self.config, self.times, self.frequency,
                                                   channel_bandwidth=self.channel_bandwidth,
                                                   phasecentre=self.phasecentre, weight=1.0,
                                                   polarisation_frame=PolarisationFrame('stokesI'),
                                                   integration_time=30.0, number_integrations=3, model=model,
                                                   components=comp)
        
        fullvis = None
        totalnvis = 0
        for i, bvis in enumerate(vis_iter):
            assert bvis.phasecentre == self.phasecentre
            assert bvis.nvis
            if i == 0:
                fullvis = bvis
                totalnvis = bvis.nvis
            else:
                fullvis = append_visibility(fullvis, bvis)
                totalnvis += bvis.nvis
        
        assert fullvis.nvis == totalnvis
    
    def test_predict_sky_components_coalesce(self):
        sc = create_low_test_skycomponents_from_gleam(flux_limit=10.0,
                                                      polarisation_frame=PolarisationFrame("stokesI"),
                                                      frequency=self.frequency, kind='cubic',
                                                      phasecentre=SkyCoord("17h20m31s", "-00d58m45s"),
                                                      radius=0.1)
        self.config = create_named_configuration('LOWBD2-CORE')
        self.phasecentre = SkyCoord("17h20m31s", "-00d58m45s")
        sampling_time = 3.76
        self.times = numpy.arange(0.0, + 300 * sampling_time, sampling_time)
        self.vis = create_blockvisibility(self.config, self.times, self.frequency, phasecentre=self.phasecentre,
                                          weight=1.0, polarisation_frame=PolarisationFrame('stokesI'),
                                          channel_bandwidth=self.channel_bandwidth)
        self.vis = predict_skycomponent_visibility(self.vis, sc)
        cvt = convert_blockvisibility_to_visibility(self.vis)
        assert cvt.cindex is not None


if __name__ == '__main__':
    unittest.main()
