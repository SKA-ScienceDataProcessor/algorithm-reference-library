""" Unit tests for pipelines expressed via dask.delayed


"""

import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from iterators import image_raster_iter
from processing_components.imaging.base import predict_skycomponent_visibility
from processing_components.skycomponent.operations import create_skycomponent
from processing_components.util.testing_support import create_named_configuration, create_test_image
from processing_components.visibility.base import create_blockvisibility

from processing_components.component_support.arlexecute import arlexecute
from processing_components.component_support.generic_components import generic_blockvisibility_component, \
    generic_image_component, \
    generic_image_iterator_component


class TestPipelinesGenericDask(unittest.TestCase):
    
    def setUp(self):
        
        arlexecute.set_client(use_dask=True)
        
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = numpy.linspace(-3, +3, 13) * (numpy.pi / 12.0)
        
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e7])
        
        # Define the component and give it some polarisation and spectral behaviour
        f = numpy.array([100.0])
        self.flux = numpy.array([f])
        
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox='J2000')
        
        self.comp = create_skycomponent(direction=self.compabsdirection, flux=self.flux, frequency=self.frequency,
                                        polarisation_frame=PolarisationFrame('stokesI'))
        self.image = create_test_image(frequency=self.frequency, phasecentre=self.phasecentre,
                                       cellsize=0.001,
                                       polarisation_frame=PolarisationFrame('stokesI'))
        self.image.data[self.image.data < 0.0] = 0.0
        
        self.image_graph = arlexecute.execute(create_test_image)(frequency=self.frequency,
                                                                 phasecentre=self.phasecentre,
                                                                 cellsize=0.001,
                                                                 polarisation_frame=PolarisationFrame('stokesI'))
    
    def test_create_generic_blockvisibility_graph(self):
        self.blockvis = [create_blockvisibility(self.lowcore, self.times, self.frequency,
                                                phasecentre=self.phasecentre,
                                                channel_bandwidth=self.channel_bandwidth,
                                                weight=1.0,
                                                polarisation_frame=PolarisationFrame('stokesI'))]
        
        self.blockvis = generic_blockvisibility_component(predict_skycomponent_visibility,
                                                          vis_list=self.blockvis,
                                                          sc=self.comp)[0]
        
        self.blockvis = arlexecute.compute(self.blockvis, sync=True)
        arlexecute.client.close()

        assert numpy.max(numpy.abs(self.blockvis[0].vis)) > 0.0
    
    def test_create_generic_image_iterator_graph(self):
        def imagerooter(im):
            im.data = numpy.sqrt(numpy.abs(im.data))
            return im
        
        root = generic_image_iterator_component(imagerooter, self.image, image_raster_iter, facets=4)
        root = arlexecute.compute(root, sync=True)
        arlexecute.client.close()
        
        numpy.testing.assert_array_almost_equal_nulp(root.data ** 2, numpy.abs(self.image.data), 7)
    
    def test_create_generic_image_graph(self):
        def imagerooter(im):
            im.data = numpy.sqrt(numpy.abs(im.data))
            return im
        
        root = generic_image_component(imagerooter, self.image, facets=4)
        root = arlexecute.compute(root, sync=True)
        arlexecute.client.close()

        numpy.testing.assert_array_almost_equal_nulp(root.data ** 2, numpy.abs(self.image.data), 7)


if __name__ == '__main__':
    unittest.main()
