"""Unit tests for visibility operations

realtimcornwell@gmail.com
"""

import unittest

from arl.util.testing_support import create_low_test_image, create_named_configuration, create_test_image, \
    create_low_test_beam, create_visibility_iterator
from arl.visibility.iterators import *
from arl.visibility.operations import create_visibility, append_visibility

import logging
log = logging.getLogger(__name__)


class TestTesting_Support(unittest.TestCase):
    def setUp(self):
        self.frequency = numpy.linspace(1e8, 1.5e8, 3)
        self.flux = numpy.array([[100.0], [100.0], [100.0]])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.config = create_named_configuration('LOWBD2-CORE')
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        nants = self.config.xyz.shape[0]
        assert nants > 1
        assert len(self.config.names) == nants
        assert len(self.config.mount) == nants

    def createVis(self, config, dec = -35.0):
        self.config = create_named_configuration(config)
        self.phasecentre = SkyCoord(ra=+15 * u.deg, dec=dec * u.deg, frame='icrs', equinox=2000.0)
        self.vis = create_visibility(self.config, self.times, self.frequency, phasecentre=self.phasecentre, weight=1.0,
                                     npol=1)

    def test_named_configurations(self):
        for config in ['LOWBD2', 'LOWBD2-CORE', 'LOWBD1', 'LOFAR']:
            self.createVis(config)
    
        self.createVis('VLAA', +35.0)
        self.createVis('VLAA_north', +35.0)

    def test_unknown_configuration(self):
        with self.assertRaises(UserWarning):
            self.config = create_named_configuration("SKA1-OWL")

    def test_create_test_image(self):
        im = create_test_image(canonical=False)
        assert len(im.data.shape) == 2
        im = create_test_image(canonical=True)
        assert len(im.data.shape) == 4
        im = create_test_image(canonical=True, npol=4, frequency=numpy.array([1e8]))
        assert len(im.data.shape) == 4
        assert im.data.shape[0] == 1
        assert im.data.shape[1] == 4

    def test_create_low_test_image(self):
        im = create_low_test_image(npixel=1024, channelwidth=1e5,
                                   frequency=numpy.array([1e8]),
                                   phasecentre=self.phasecentre, fov=10)
        assert im.data.shape[0] == 1
        assert im.data.shape[1] == 1
        assert im.data.shape[2] == 1024
        assert im.data.shape[3] == 1024

    def test_create_low_test_image_no_phasecentre(self):
    
        im = create_low_test_image(npixel=1024, channelwidth=1e5,
                                   frequency=numpy.array([1e8]),
                                   fov=10)
        assert im.data.shape[0] == 1
        assert im.data.shape[1] == 1
        assert im.data.shape[2] == 1024
        assert im.data.shape[3] == 1024

    def test_create_low_test_beam(self):
        im = create_test_image(canonical=True, npol=1, frequency=numpy.array([1e8]), phasecentre=self.phasecentre)
        bm = create_low_test_beam(model=im)
        assert bm.data.shape[0] == 1
        assert bm.data.shape[1] == 1
        assert bm.data.shape[2] == im.data.shape[2]
        assert bm.data.shape[3] == im.data.shape[3]
        
    def test_create_vis_iter(self):
        vis_iter = create_visibility_iterator(self.config, self.times, self.frequency, phasecentre=self.phasecentre,
                                              weight=1.0, npol=1, integration_time=30.0, number_integrations=3)

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
        model = create_test_image(canonical=True, cellsize=0.001, npol=1, frequency=self.frequency,
                                  phasecentre=self.phasecentre)
        comp=Skycomponent(direction=self.phasecentre, frequency=self.frequency, flux=self.flux)
        vis_iter = create_visibility_iterator(self.config, self.times, self.frequency, phasecentre=self.phasecentre,
                                              weight=1.0, npol=1, integration_time=30.0, number_integrations=3,
                                              model=model, components=comp)

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