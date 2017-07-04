"""Unit tests for testing support


"""

import os
import unittest

from arl.fourier_transforms.ftprocessor_base import predict_skycomponent_blockvisibility
from arl.image.operations import export_image_to_fits
from arl.skycomponent.operations import apply_beam_to_skycomponent
from arl.util.testing_support import create_low_test_image_from_s3, create_named_configuration, create_test_image, \
    create_low_test_beam, create_blockvisibility_iterator, create_low_test_image_from_gleam, \
    create_low_test_skycomponents_from_gleam, create_low_test_image_composite
from arl.visibility.iterators import *
from arl.visibility.operations import create_blockvisibility, create_visibility, \
    append_visibility, copy_visibility
from arl.visibility.coalesce import coalesce_visibility
from arl.util.dask_graph_support import create_simulate_vis_graph, create_corrupt_vis_graph, \
    create_load_vis_graph, create_dump_vis_graph

log = logging.getLogger(__name__)


class TestTestingDaskGraphSupport(unittest.TestCase):
    def setUp(self):
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        
        self.frequency = numpy.linspace(1e8, 1.5e8, 3)
        self.channel_bandwidth = numpy.array([2.5e7, 2.5e7, 2.5e7])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox=2000.0)
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        
    def test_create_simulate_vis_grap(self):
        vis_graph_list = create_simulate_vis_graph(frequency=self.frequency, channel_bandwidth=self.channel_bandwidth)
        assert len(vis_graph_list) == len(self.frequency)
        
    def test_corrupt_vis_graph(self):
        vis_graph_list = create_simulate_vis_graph(frequency=self.frequency, channel_bandwidth=self.channel_bandwidth)
        corrupt_vis_graph_list = create_corrupt_vis_graph(vis_graph_list, phase_error=0.1)
        assert len(corrupt_vis_graph_list) == len(vis_graph_list)

    def test_dump_load_graph(self):
        vis_graph_list = create_simulate_vis_graph(frequency=self.frequency, channel_bandwidth=self.channel_bandwidth)
        dump_graph=create_dump_vis_graph(vis_graph_list, name='test_results/imaging_dask_%d.pickle')
        load_graph=create_load_vis_graph(name='test_results/imaging_dask_*.pickle')
        assert len(load_graph) == len(vis_graph_list)