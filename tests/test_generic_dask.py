"""Unit tests for pipelines expressed via dask.delayed


"""

import unittest

from dask import delayed

from arl.pipelines.generic_dask_graphs import create_generic_blockvisibility_graph, create_generic_image_graph
from arl.pipelines.dask_graphs import create_predict_graph
from arl.pipelines.functions import *
from arl.skycomponent.operations import create_skycomponent
from arl.util.testing_support import create_named_configuration, create_test_image, simulate_gaintable


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
        
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox=2000.0)
        
        self.comp = create_skycomponent(flux=self.flux, frequency=self.frequency, direction=self.compabsdirection,
                                        polarisation_frame=PolarisationFrame('stokesI'))
        self.image = create_test_image(frequency=self.frequency, phasecentre=self.phasecentre,
                                       cellsize=0.001,
                                       polarisation_frame=PolarisationFrame('stokesI'))

        self.image_graph = delayed(create_test_image)(frequency=self.frequency, phasecentre=self.phasecentre,
                                                      cellsize=0.001, polarisation_frame=PolarisationFrame('stokesI'))

        self.vis = create_visibility(self.lowcore, times=self.times, frequency=self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1,
                                     polarisation_frame=PolarisationFrame('stokesI'),
                                     integration_time=1.0)
        
    
    def test_create_generic_blockvisibility_graph(self):
        self.blockvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                               phasecentre=self.phasecentre, weight=1.0,
                                               polarisation_frame=PolarisationFrame('stokesI'))
        self.blockvis = \
            create_generic_blockvisibility_graph(predict_skycomponent_blockvisibility)(self.blockvis,
                                                                                       vis_timeslice_iter,
                                                                                       sc=self.comp).compute()
        
        assert numpy.max(numpy.abs(self.blockvis.vis)) > 0.0

    def test_create_generic_image_graph(self):

        def imagerooter(im, **kwargs):
            im.data = numpy.sqrt(numpy.abs(im.data))
            return im
        root_graph = create_generic_image_graph(imagerooter)
        root = root_graph(self.image, raster_iter, facets=2).compute()
        numpy.testing.assert_array_almost_equal_nulp(root.data**2, numpy.abs(self.image.data), 7)
    

if __name__ == '__main__':
    unittest.main()
