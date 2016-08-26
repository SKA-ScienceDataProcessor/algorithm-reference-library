import unittest

import numpy
from numpy.testing import assert_allclose

from arl.image_operations import *


class TestImage(unittest.TestCase):

    def setUp(self):
        self.m31image = replicate_image(create_image_from_fits("./data/models/M31.MOD"))
        self.cellsize = 180.0 * 0.0001 / numpy.pi
        self.m31image.wcs.wcs.cdelt[0] = -self.cellsize
        self.m31image.wcs.wcs.cdelt[1] = +self.cellsize
        self.m31image.wcs.wcs.radesys = 'ICRS'
        self.m31image.wcs.wcs.equinox = 2000.00

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
        save_image_to_fits(newimage, fitsfile='reproject_image.fits')
        save_image_to_fits(footprint, fitsfile='reproject_footprint.fits')

if __name__ == '__main__':
    unittest.main()
