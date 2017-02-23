"""Unit tests for image operations

realtimcornwell@gmail.com
"""
import sys
import unittest

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
    
        m31model_by_array = create_image_from_array(self.m31image.data, self.m31image.wcs)
        # noinspection PyBroadException
        try:
            m31modelsum = add_image(self.m31image, m31model_by_array)
        except:
            log.debug("Image: correctly failed on checkwcs=True")
            pass
        m31modelsum = add_image(self.m31image, m31model_by_array)
        log.debug(self.m31image.data.shape)
        log.debug(self.m31image.wcs)
        log.debug(export_image_to_fits(self.m31image, fitsfile='%s/test_model.fits' % (self.dir)))

    def test_reproject(self):
        # Reproject an image
        
        cellsize=1.5*self.cellsize
        newwcs=self.m31image.wcs.deepcopy()
        newwcs.wcs.cdelt[0] = -cellsize
        newwcs.wcs.cdelt[1] = +cellsize
        
        newshape=numpy.array(self.m31image.data.shape)
        newshape[2] /= 1.5
        newshape[3] /= 1.5
        newimage, footprint=reproject_image(self.m31image, newwcs, shape=newshape)
        checkwcs(newimage.wcs, newwcs)
        
        # export_image_to_fits(newimage, fitsfile='%s/reproject_image.fits' % (self.dir))
        # export_image_to_fits(footprint, fitsfile='%s/reproject_footprint.fits' % (self.dir))

if __name__ == '__main__':
    run_unittests()
