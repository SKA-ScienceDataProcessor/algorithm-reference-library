"""Unit tests for image iteration


"""
import logging
import unittest

import numpy

from arl.data.polarisation import PolarisationFrame
from arl.image.iterators import   image_raster_iter, image_channel_iter, image_null_iter
from arl.image.operations import create_empty_image_like
from arl.util.testing_support import create_test_image

log = logging.getLogger(__name__)


class TestImageIterators(unittest.TestCase):
    def test_raster(self):
        
        m31original = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
        assert numpy.max(numpy.abs(m31original.data)), "Original is empty"
        
        for nraster in [1, 2, 4, 8, 9]:
            m31model = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
            for patch in image_raster_iter(m31model, facets=nraster):
                assert patch.data.shape[3] == (m31model.data.shape[3] // nraster), \
                    "Number of pixels in each patch: %d not as expected: %d" % (patch.data.shape[3],
                                                                                (m31model.data.shape[3] // nraster))
                assert patch.data.shape[2] == (m31model.data.shape[2] // nraster), \
                    "Number of pixels in each patch: %d not as expected: %d" % (patch.data.shape[2],
                                                                                (m31model.data.shape[2] // nraster))
                patch.data *= 2.0
            
            diff = m31model.data - 2.0 * m31original.data
            assert numpy.max(numpy.abs(m31model.data)), "Raster is empty for %d" % nraster
            assert numpy.max(numpy.abs(diff)) == 0.0, "Raster set failed for %d" % nraster

    def test_raster_overlap(self):
    
        m31original = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
        assert numpy.max(numpy.abs(m31original.data)), "Original is empty"
        flat = create_empty_image_like(m31original)
    
        for nraster, overlap in [(1, 0),  (1, 16), (4, 8), (4, 16), (8, 8), (16, 4), (9, 5)]:
            m31model = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
            for patch, flat_patch in zip(image_raster_iter(m31model, facets=nraster, overlap=overlap),
                                         image_raster_iter(flat, facets=nraster, overlap=overlap)):
                patch.data *= 2.0
                flat_patch.data[...] += 1.0
        
            assert numpy.max(numpy.abs(m31model.data)), "Raster is empty for %d" % nraster

    def test_raster_overlap_linear(self):
    
        m31original = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
        assert numpy.max(numpy.abs(m31original.data)), "Original is empty"
        flat = create_empty_image_like(m31original)
    
        for nraster, overlap in [(1, 0), (1, 16), (4, 8), (4, 16), (8, 8), (16, 4), (9, 5)]:
            m31model = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
            for patch, flat_patch in zip(image_raster_iter(m31model, facets=nraster, overlap=overlap,
                                                           taper='linear'),
                                         image_raster_iter(flat, facets=nraster, overlap=overlap)):
                patch.data *= 2.0
                flat_patch.data[...] += 1.0
        
            assert numpy.max(numpy.abs(m31model.data)), "Raster is empty for %d" % nraster

    def test_raster_overlap_quadratic(self):
    
        m31original = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
        assert numpy.max(numpy.abs(m31original.data)), "Original is empty"
        flat = create_empty_image_like(m31original)
    
        for nraster, overlap in [(1, 0), (1, 16), (4, 8), (4, 16), (8, 8), (16, 4), (9, 5)]:
            m31model = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
            for patch, flat_patch in zip(image_raster_iter(m31model, facets=nraster, overlap=overlap,
                                                           taper='quadratic'),
                                         image_raster_iter(flat, facets=nraster, overlap=overlap)):
                patch.data *= 2.0
                flat_patch.data[...] += 1.0
        
            assert numpy.max(numpy.abs(m31model.data)), "Raster is empty for %d" % nraster

    def test_raster_overlap_tukey(self):
    
        m31original = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
        assert numpy.max(numpy.abs(m31original.data)), "Original is empty"
        flat = create_empty_image_like(m31original)
    
        for nraster, overlap in [(1, 0), (1, 16), (4, 8), (4, 16), (8, 8), (16, 4), (9, 5)]:
            m31model = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
            for patch, flat_patch in zip(image_raster_iter(m31model, facets=nraster, overlap=overlap,
                                                           taper='tukey'),
                                         image_raster_iter(flat, facets=nraster, overlap=overlap)):
                patch.data *= 2.0
                flat_patch.data[...] += 1.0
        
            assert numpy.max(numpy.abs(m31model.data)), "Raster is empty for %d" % nraster

    def test_channelise(self):
        m31cube = create_test_image(polarisation_frame=PolarisationFrame('stokesI'),
                                        frequency=numpy.linspace(1e8,1.1e8, 128))
        
        for subimages in [128, 16, 8, 2, 1]:
            for slab in image_channel_iter(m31cube, subimages=subimages):
                assert slab.data.shape[0] == 128 // subimages

    def test_null(self):
        m31cube = create_test_image(polarisation_frame=PolarisationFrame('stokesI'),
                                    frequency=numpy.linspace(1e8, 1.1e8, 128))
    
        for i, im in enumerate(image_null_iter(m31cube)):
            assert i<1, "Null iterator returns more than one value"


if __name__ == '__main__':
    unittest.main()
