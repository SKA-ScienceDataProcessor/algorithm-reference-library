"""Unit tests for image operations

realtimcornwell@gmail.com
"""
import sys
import unittest

import numpy

from arl.data.polarisation import Polarisation_Frame
from arl.image.iterators import *
from arl.util.testing_support import create_test_image
from arl.util.run_unittests import run_unittests


log = logging.getLogger(__name__)


class TestImageIterators(unittest.TestCase):
    
    def test_rasterise(self):
    
        m31original = create_test_image(polarisation_frame=Polarisation_Frame('stokesI'))
        assert numpy.max(numpy.abs(m31original.data)), "Original is empty"

        params = {}
        for nraster in [2, 4, 8]:
            m31model = create_test_image(polarisation_frame=Polarisation_Frame('stokesI'))
            for patch in raster_iter(m31model, facets=nraster):
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


if __name__ == '__main__':
    run_unittests()
