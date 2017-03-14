"""Unit tests for image operations

realtimcornwell@gmail.com
"""
import sys
import unittest

import numpy
from numpy.testing import *

from arl.image.iterators import *
from arl.image.operations import *
from arl.util.testing_support import create_test_image
from arl.util.run_unittests import run_unittests


log = logging.getLogger(__name__)

class TestImage(unittest.TestCase):

    def setUp(self):
    
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
    
        self.m31image = create_test_image(cellsize=0.0001)
        self.cellsize = 180.0 * 0.0001 / numpy.pi

    def test_create_image_from_array(self):
        m31model_by_array = create_image_from_array(self.m31image.data, wcs=None)
        
        m31model_by_array = create_image_from_array(self.m31image.data, self.m31image.wcs)
        m31modelsum = add_image(self.m31image, m31model_by_array)
        m31modelsum = add_image(self.m31image, m31model_by_array, docheckwcs=True)
        assert m31model_by_array.shape == self.m31image.shape
        log.debug(export_image_to_fits(self.m31image, fitsfile='%s/test_model.fits' % (self.dir)))
        log.debug(qa_image(m31model_by_array, context='test_create_from_image'))

    def test_create_empty_image_like(self):
        emptyimage = create_empty_image_like(self.m31image)
        assert emptyimage.shape == self.m31image.shape
        assert numpy.max(numpy.abs(emptyimage.data)) == 0.0

    def test_checkwcs(self):
    
        newwcs = self.m31image.wcs.deepcopy()
        newwcs = self.m31image.wcs
        checkwcs(self.m31image.wcs, newwcs)
        cellsize = 1.5 * self.cellsize
        newwcs.wcs.cdelt[0] = -cellsize
        newwcs.wcs.cdelt[1] = +cellsize
        # with self.assertRaises(AssertionError):
        #     checkwcs(self.m31image.wcs, newwcs)
    
    def test_reproject(self):
        # Reproject an image
    
        cellsize = 1.5 * self.cellsize
        newwcs = self.m31image.wcs.deepcopy()
        newwcs.wcs.cdelt[0] = -cellsize
        newwcs.wcs.cdelt[1] = +cellsize
    
        newshape = numpy.array(self.m31image.data.shape)
        newshape[2] /= 1.5
        newshape[3] /= 1.5
        newimage, footprint = reproject_image(self.m31image, newwcs, shape=newshape)
        checkwcs(newimage.wcs, newwcs)

    def test_stokes_conversion(self):
        assert self.m31image.polarisation_frame == PolarisationFrame("stokesI")
        stokes = create_test_image(cellsize=0.0001, polarisation_frame=PolarisationFrame("stokesIQUV"))
        assert stokes.polarisation_frame == PolarisationFrame("stokesIQUV")

        for pol_name in ['circular', 'linear']:
            polarisation_frame = PolarisationFrame(pol_name)
            polimage = convert_stokes_to_polimage(stokes, polarisation_frame=polarisation_frame)
            assert polimage.polarisation_frame == polarisation_frame
            pf = polarisation_frame_from_wcs(polimage.wcs, polimage.shape)
            rstokes = convert_polimage_to_stokes(polimage)
            assert polimage.data.dtype == 'complex'
            assert rstokes.data.dtype == 'complex'
            numpy.testing.assert_array_almost_equal(stokes.data, rstokes.data.real, 12)
            
    def test_polarisation_frame_from_wcs(self):
        assert self.m31image.polarisation_frame == PolarisationFrame("stokesI")
        stokes = create_test_image(cellsize=0.0001, polarisation_frame=PolarisationFrame("stokesIQUV"))
        wcs = stokes.wcs.deepcopy()
        shape = stokes.shape
        assert polarisation_frame_from_wcs(wcs, shape) == PolarisationFrame("stokesIQUV")
        
        wcs = stokes.wcs.deepcopy().sub(['stokes'])
        wcs.wcs.crpix[0] = 1.0
        wcs.wcs.crval[0] = -1.0
        wcs.wcs.cdelt[0] = -1.0
        assert polarisation_frame_from_wcs(wcs, shape) == PolarisationFrame('circular')

        wcs.wcs.crpix[0] = 1.0
        wcs.wcs.crval[0] = -5.0
        wcs.wcs.cdelt[0] = -1.0
        assert polarisation_frame_from_wcs(wcs, shape) == PolarisationFrame('linear')

        wcs.wcs.crpix[0] = 1.0
        wcs.wcs.crval[0] = -1.0
        wcs.wcs.cdelt[0] = -1.0
        assert polarisation_frame_from_wcs(wcs, shape) == PolarisationFrame('circular')

        with self.assertRaises(ValueError):
            wcs.wcs.crpix[0] = 1.0
            wcs.wcs.crval[0] = -100.0
            wcs.wcs.cdelt[0] = -1.0
            pf = polarisation_frame_from_wcs(wcs, shape)

    def test_show_image(self):
        show_image(self.m31image)

if __name__ == '__main__':
    run_unittests()
