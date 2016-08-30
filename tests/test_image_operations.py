"""Unit tests for image operations

realtimcornwell@gmail.com
"""
import unittest

import os
import numpy
from numpy.testing import assert_allclose

from arl.image_operations import *
from arl.test_support import replicate_image


class TestImage(unittest.TestCase):

    def setUp(self):
        chome = os.environ['CROCODILE']
        kwargs = {}
        self.m31image = replicate_image(import_image_from_fits("%s/data/models/M31.MOD" % chome))
        self.cellsize = 180.0 * 0.0001 / numpy.pi
        self.m31image.wcs.wcs.cdelt[0] = -self.cellsize
        self.m31image.wcs.wcs.cdelt[1] = +self.cellsize
        self.m31image.wcs.wcs.radesys = 'ICRS'
        self.m31image.wcs.wcs.equinox = 2000.00
        
    def test_create_image_from_array(self):
    
        m31model_by_array = create_image_from_array(self.m31image.data, self.m31image.wcs)
        try:
            m31modelsum = add_image(self.m31image, m31model_by_array, checkwcs=True)
        except:
            print("Image: correctly failed on checkwcs=True")
            pass
        m31modelsum = add_image(self.m31image, m31model_by_array)
        print(self.m31image.data.shape)
        print(self.m31image.wcs)
        print(export_image_to_fits(self.m31image, fitsfile='temp.fits'))

    def test_reproject(self):
        # Reproject an image
        
        cellsize=1.5*self.cellsize
        newwcs=self.m31image.wcs
        newwcs.wcs.cdelt[0] = -cellsize
        newwcs.wcs.cdelt[1] = +cellsize
        newwcs.wcs.radesys = 'ICRS'
        newwcs.wcs.equinox = 2000.00
        
        newshape=(1,1,int(256//1.5),int(256//1.5))
        newimage, footprint=reproject_image(self.m31image, newwcs, shape=newshape)
        export_image_to_fits(newimage, fitsfile='reproject_image.fits')
        export_image_to_fits(footprint, fitsfile='reproject_footprint.fits')

if __name__ == '__main__':
    unittest.main()
