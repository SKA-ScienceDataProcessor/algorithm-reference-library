"""Unit tests for pipelines expressed via dask.delayed


"""

import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from dask import delayed

from arl.data.polarisation import PolarisationFrame
from arl.graphs.generic_graphs import create_generic_blockvisibility_graph, create_generic_image_graph, \
    create_generic_image_iterator_graph
from arl.image.iterators import raster_iter
from arl.imaging import predict_skycomponent_blockvisibility
from arl.skycomponent.operations import create_skycomponent
from arl.util.testing_support import create_named_configuration, create_test_image
from arl.visibility.base import create_blockvisibility


class TestPipelinesGenericDask(unittest.TestCase):
    def setUp(self):
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = numpy.linspace(-3, +3, 13) * (numpy.pi / 12.0)
        
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e7])
        
        # Define the component and give it some polarisation and spectral behaviour
        f = numpy.array([100.0])
        self.flux = numpy.array([f])
        
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox='J2000')
        
        self.comp = create_skycomponent(flux=self.flux, frequency=self.frequency, direction=self.compabsdirection,
                                        polarisation_frame=PolarisationFrame('stokesI'))
        self.image = create_test_image(frequency=self.frequency, phasecentre=self.phasecentre,
                                       cellsize=0.001,
                                       polarisation_frame=PolarisationFrame('stokesI'))
        self.image.data[self.image.data<0.0]=0.0

        self.image_graph = delayed(create_test_image)(frequency=self.frequency, phasecentre=self.phasecentre,
                                                      cellsize=0.001, polarisation_frame=PolarisationFrame('stokesI'))
    
    def test_create_generic_blockvisibility_graph(self):
        self.blockvis = [create_blockvisibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                          channel_bandwidth=self.channel_bandwidth,
                                          weight=1.0, polarisation_frame=PolarisationFrame('stokesI'))]
        self.blockvis = \
            create_generic_blockvisibility_graph(predict_skycomponent_blockvisibility, vis_graph_list=self.blockvis,
                                                 sc=self.comp)[0].compute()
        
        assert numpy.max(numpy.abs(self.blockvis[0].vis)) > 0.0

    def test_create_generic_image_iterator_graph(self):
        def imagerooter(im, **kwargs):
            im.data = numpy.sqrt(numpy.abs(im.data))
            return im
    
        root = create_generic_image_iterator_graph(imagerooter, self.image, raster_iter, facets=16).compute()
        numpy.testing.assert_array_almost_equal_nulp(root.data ** 2, numpy.abs(self.image.data), 7)

    def test_create_generic_image_graph(self):
        def imagerooter(im, **kwargs):
            im.data = numpy.sqrt(numpy.abs(im.data))
            return im
    
        root = create_generic_image_graph(imagerooter, self.image, facets=16).compute()
        numpy.testing.assert_array_almost_equal_nulp(root.data ** 2, numpy.abs(self.image.data), 7)


if __name__ == '__main__':
    unittest.main()
