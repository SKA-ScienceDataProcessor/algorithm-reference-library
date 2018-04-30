"""Unit tests for image iteration


"""
import logging
import unittest

import numpy

from data_models.polarisation import PolarisationFrame
from processing_components.image.operations import create_empty_image_like, export_image_to_fits
from processing_components.image.gather_scatter import image_gather_facets, image_scatter_facets, image_gather_channels, \
    image_scatter_channels
from processing_components.util.testing_support import create_test_image

log = logging.getLogger(__name__)


class TestImageGatherScatters(unittest.TestCase):
    
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')

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
            flat = image_gather_facets(image_list, m31reconstructed, facets=nraster, return_flat=True)

            assert numpy.max(numpy.abs(flat.data)), "Flat is empty for %d" % nraster
            assert numpy.max(numpy.abs(m31reconstructed.data)), "Raster is empty for %d" % nraster

    def test_scatter_gather_facet_overlap(self):
    
        m31original = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
        assert numpy.max(numpy.abs(m31original.data)), "Original is empty"
    
        for nraster, overlap in [(1, 0), (4, 8), (8, 16)]:
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
            flat = image_gather_facets(image_list, m31reconstructed, facets=nraster, overlap=overlap, return_flat=True)
        
            assert numpy.max(numpy.abs(flat.data)), "Flat is empty for %d" % nraster
            assert numpy.max(numpy.abs(m31reconstructed.data)), "Raster is empty for %d" % nraster

    def test_scatter_gather_facet_overlap_taper(self):
    
        m31original = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
        assert numpy.max(numpy.abs(m31original.data)), "Original is empty"
    
        for taper in ['linear', None]:
            for nraster, overlap in [(1, 0), (4, 8), (8, 8), (8, 16)]:
                m31model = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
                image_list = image_scatter_facets(m31model, facets=nraster, overlap=overlap, taper=taper)
                for patch in image_list:
                    assert patch.data.shape[3] == (2 * overlap + m31model.data.shape[3] // nraster), \
                        "Number of pixels in each patch: %d not as expected: %d" % (patch.data.shape[3],
                                                                                    (2 * overlap + m31model.data.shape[3] //
                                                                                     nraster))
                    assert patch.data.shape[2] == (2 * overlap + m31model.data.shape[2] // nraster), \
                        "Number of pixels in each patch: %d not as expected: %d" % (patch.data.shape[2],
                                                                                    (2 * overlap + m31model.data.shape[2] //
                                                                                     nraster))
                m31reconstructed = create_empty_image_like(m31model)
                m31reconstructed = image_gather_facets(image_list, m31reconstructed, facets=nraster, overlap=overlap,
                                                       taper=taper)
                flat = image_gather_facets(image_list, m31reconstructed, facets=nraster, overlap=overlap,
                                           taper=taper, return_flat=True)
                export_image_to_fits(m31reconstructed,
                                     "%s/test_image_gather_scatter_%dnraster_%doverlap_%s_reconstructed.fits" %
                                     (self.dir, nraster, overlap, taper))
                export_image_to_fits(flat,
                                     "%s/test_image_gather_scatter_%dnraster_%doverlap_%s_flat.fits" %
                                     (self.dir, nraster, overlap, taper))
    
                assert numpy.max(numpy.abs(flat.data)), "Flat is empty for %d" % nraster
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
