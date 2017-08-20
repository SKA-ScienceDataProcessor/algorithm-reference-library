"""Unit tests for testing support


"""

import numpy
import os
import unittest

from astropy.coordinates import SkyCoord
import astropy.units as u

from arl.data.data_models import Skycomponent
from arl.data.polarisation import PolarisationFrame
from arl.image.operations import export_image_to_fits
from arl.skycomponent.operations import apply_beam_to_skycomponent
from arl.util.testing_support import create_low_test_image_from_s3, create_named_configuration, create_test_image, \
    create_low_test_beam, create_blockvisibility_iterator, create_low_test_image_from_gleam, \
    create_low_test_skycomponents_from_gleam, create_low_test_image_composite
from arl.visibility.coalesce import coalesce_visibility
from arl.visibility.operations import append_visibility
from arl.visibility.base import create_visibility, create_blockvisibility
from arl.imaging.base import predict_skycomponent_blockvisibility

import logging
log = logging.getLogger(__name__)


class TestTesting_Support(unittest.TestCase):
    def setUp(self):
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        
        self.frequency = numpy.linspace(1e8, 1.5e8, 3)
        self.channel_bandwidth = numpy.array([2.5e7, 2.5e7, 2.5e7])
        self.flux = numpy.array([[100.0], [100.0], [100.0]])
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

    def test_named_configurations(self):
        for config in ['LOWBD2', 'LOWBD2-CORE', 'LOWBD1', 'LOFAR']:
            self.createVis(config)
            assert self.config.size() > 0.0
    
        self.createVis('VLAA', +35.0)
        self.createVis('VLAA_north', +35.0)

    def test_clip_configuration(self):
        for rmax in [100.0, 3000.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]:
            self.config = create_named_configuration('LOWBD2', rmax=rmax)
            assert self.config.size() > 0.0
    
    def test_unknown_configuration(self):
        with self.assertRaises(ValueError):
            self.config = create_named_configuration("SKA1-OWL")
    
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
    
    def test_create_low_test_image_from_gleam(self):
        im = create_low_test_image_from_gleam(npixel=256, cellsize=0.001,
                                              channel_bandwidth=self.channel_bandwidth,
                                              frequency=self.frequency,
                                              phasecentre=self.phasecentre, kind='cubic')
        assert im.data.shape[0] == 3
        assert im.data.shape[1] == 1
        assert im.data.shape[2] == 256
        assert im.data.shape[3] == 256
        export_image_to_fits(im, '%s/test_low_gleam.fits' % (self.dir))
    
    def test_create_low_test_image_composite(self):
        im = create_low_test_image_composite(npixel=256, cellsize=0.001,
                                             channel_bandwidth=self.channel_bandwidth,
                                             frequency=self.frequency,
                                             phasecentre=self.phasecentre, kind='cubic',
                                             threshold=0.050, fov=20)
        assert im.data.shape[0] == 3
        assert im.data.shape[1] == 1
        assert im.data.shape[2] == 256
        assert im.data.shape[3] == 256
        export_image_to_fits(im, '%s/test_low_composite.fits' % (self.dir))
    
    def test_create_low_test_skycomponents_from_gleam_apply_beam(self):
        sc = create_low_test_skycomponents_from_gleam(flux_limit=10.0,
                                                      polarisation_frame=PolarisationFrame("stokesI"),
                                                      frequency=self.frequency, kind='cubic')
        assert len(sc) > 1
        assert sc[190].name == 'GLEAM J172031-005845'
        #        self.assertAlmostEqual(sc[190].flux[0,0], 301.4964434927922, 7)
        im = create_test_image(canonical=True, cellsize=0.002,
                               frequency=self.frequency,
                               channel_bandwidth=self.channel_bandwidth,
                               polarisation_frame=PolarisationFrame("stokesI"),
                               phasecentre=self.phasecentre)
        bm = create_low_test_beam(model=im)
        sc = apply_beam_to_skycomponent(sc, bm)
        assert len(sc) > 1, "No components inside image"
    
    def test_create_low_test_skycomponents_from_gleam_apply_filter(self):
        sc = create_low_test_skycomponents_from_gleam(flux_limit=10.0,
                                                      polarisation_frame=PolarisationFrame("stokesI"),
                                                      frequency=self.frequency, kind='cubic',
                                                      phasecentre=SkyCoord("17h20m31s", "-00d58m45s"),
                                                      radius=0.1)
        assert len(sc) == 1
        assert sc[0].name == 'GLEAM J172031-005845'
    
    def test_create_low_test_image(self):
        im = create_low_test_image_from_s3(npixel=1024, channel_bandwidth=numpy.array([1e6]),
                                           frequency=numpy.array([1e8]),
                                           phasecentre=self.phasecentre, fov=10)
        assert im.data.shape[0] == 1
        assert im.data.shape[1] == 1
        assert im.data.shape[2] == 1024
        assert im.data.shape[3] == 1024
    
    def test_create_low_test_image_spectral(self):
        im = create_low_test_image_from_s3(npixel=1024, channel_bandwidth=numpy.array([1e6, 1e6, 1e6]),
                                           frequency=numpy.array([1e8 - 1e6, 1e8, 1e8 + 1e6]),
                                           phasecentre=self.phasecentre, fov=10)
        assert im.data.shape[0] == 3
        assert im.data.shape[1] == 1
        assert im.data.shape[2] == 1024
        assert im.data.shape[3] == 1024
    
    def test_create_low_test_image_spectral_polarisation(self):
        
        im = create_low_test_image_from_s3(npixel=1024, channel_bandwidth=numpy.array([1e6, 1e6, 1e6]),
                                           polarisation_frame=PolarisationFrame("stokesIQUV"),
                                           frequency=numpy.array([1e8 - 1e6, 1e8, 1e8 + 1e6]), fov=10)
        assert im.data.shape[0] == 3
        assert im.data.shape[1] == 4
        assert im.data.shape[2] == 1024
        assert im.data.shape[3] == 1024
        export_image_to_fits(im, '%s/test_low_s3.fits' % (self.dir))
    
    def test_create_low_test_beam(self):
        im = create_test_image(canonical=True, cellsize=0.002,
                               frequency=numpy.array([1e8 - 5e7, 1e8, 1e8 + 5e7]),
                               channel_bandwidth=numpy.array([5e7, 5e7, 5e7]),
                               polarisation_frame=PolarisationFrame("stokesIQUV"),
                               phasecentre=self.phasecentre)
        bm = create_low_test_beam(model=im)
        export_image_to_fits(bm, '%s/test_low_beam.fits' % (self.dir))
        
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
        for i, vis in enumerate(vis_iter):
            assert vis.phasecentre == self.phasecentre
            assert vis.nvis
            if i == 0:
                fullvis = vis
                totalnvis = vis.nvis
            else:
                fullvis = append_visibility(fullvis, vis)
                totalnvis += vis.nvis
        
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
        self.vis = predict_skycomponent_blockvisibility(self.vis, sc)
        cvt = coalesce_visibility(self.vis, time_coal=1.0)
        assert cvt.cindex is not None
