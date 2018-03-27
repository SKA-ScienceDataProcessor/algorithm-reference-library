"""Unit tests for image iteration


"""
import logging
import unittest

import numpy

from arl.data.polarisation import PolarisationFrame
from arl.image.operations import create_empty_image_like
from arl.image.gather_scatter import image_gather_facets, image_scatter_facets, image_gather_channels, \
    image_scatter_channels
from arl.util.testing_support import create_test_image

log = logging.getLogger(__name__)


class TestImageGatherScatters(unittest.TestCase):
    
    def test_scatter_gather_facet(self):
        
        m31original = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
        assert numpy.max(numpy.abs(m31original.data)), "Original is empty"
        
        for nraster in [1, 4, 8]:
            m31model = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
            image_list = image_scatter_facets(m31model, facets=nraster)
            for patch in image_list:
                assert patch.data.shape[3] == (m31model.data.shape[3] // nraster), \
                    "Number of pixels in each patch: %d not as expected: %d" % (patch.data.shape[3],
                                                                                (m31model.data.shape[3] // nraster))
                assert patch.data.shape[2] == (m31model.data.shape[2] // nraster), \
                    "Number of pixels in each patch: %d not as expected: %d" % (patch.data.shape[2],
                                                                                (m31model.data.shape[2] // nraster))
                patch.data[...] = 1.0
            m31reconstructed = create_empty_image_like(m31model)
            m31reconstructed = image_gather_facets(image_list, m31reconstructed, facets=nraster)
            
            assert numpy.max(numpy.abs(m31reconstructed.data)), "Raster is empty for %d" % nraster

    def test_scatter_gather_facet_overlap(self):
    
        m31original = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
        assert numpy.max(numpy.abs(m31original.data)), "Original is empty"
    
        for nraster, overlap in [(1, 0), (4, 8), (8, 32)]:
            m31model = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
            image_list = image_scatter_facets(m31model, facets=nraster, overlap=overlap)
            for patch in image_list:
                assert patch.data.shape[3] == (2 * overlap + m31model.data.shape[3] // nraster), \
                    "Number of pixels in each patch: %d not as expected: %d" % (patch.data.shape[3],
                                                                                (2 * overlap + m31model.data.shape[3] //
                                                                                 nraster))
                assert patch.data.shape[2] == (2 * overlap + m31model.data.shape[2] // nraster), \
                    "Number of pixels in each patch: %d not as expected: %d" % (patch.data.shape[2],
                                                                                (2 * overlap + m31model.data.shape[2] //
                                                                                 nraster))
                patch.data[...] = 1.0
            m31reconstructed = create_empty_image_like(m31model)
            m31reconstructed = image_gather_facets(image_list, m31reconstructed, facets=nraster, overlap=overlap)

            assert numpy.max(numpy.abs(m31reconstructed.data)), "Raster is empty for %d" % nraster

    def test_scatter_gather_channel(self):
        for nchan in [128, 16]:
            m31cube = create_test_image(polarisation_frame=PolarisationFrame('stokesI'),
                                        frequency=numpy.linspace(1e8, 1.1e8, nchan))
            
            for subimages in [16, 8, 2, 1]:
                image_list = image_scatter_channels(m31cube, subimages=subimages)
                m31cuberec = image_gather_channels(image_list, m31cube, subimages=subimages)
                diff = m31cube.data - m31cuberec.data
                assert numpy.max(numpy.abs(diff)) == 0.0, "Scatter gather failed for %d" % subimages
    
    def test_gather_channel(self):
        for nchan in [128, 16]:
            m31cube = create_test_image(polarisation_frame=PolarisationFrame('stokesI'),
                                        frequency=numpy.linspace(1e8, 1.1e8, nchan))
            image_list = image_scatter_channels(m31cube, subimages=nchan)
            m31cuberec = image_gather_channels(image_list, None, subimages=nchan)
            assert m31cube.shape == m31cuberec.shape
            diff = m31cube.data - m31cuberec.data
            assert numpy.max(numpy.abs(diff)) == 0.0, "Scatter gather failed for %d" % nchan


if __name__ == '__main__':
    unittest.main()
