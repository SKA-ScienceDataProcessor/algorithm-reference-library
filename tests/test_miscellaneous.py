import unittest

import numpy
from numpy.testing import assert_allclose

from functions.skycomponent import SkyComponent
from functions.configuration import named_configuration
from functions.gaintable import gaintable_from_array
from functions.image import Image, image_add, image_from_array, image_from_fits
from functions.skymodel import SkyModel, skymodel_from_image, skymodel_add_component
from functions.visibility import Visibility, visibility_from_configuration

from astropy.coordinates import SkyCoord


class TestFunctions(unittest.TestCase):
    def test_wcs_from_visibility(self):
        config = named_configuration('VLAA')
        times = numpy.arange(-3.0, +3.0, 3.0 / 60.0) * numpy.pi / 12.0
        freq = numpy.arange(5e6, 150.0e6, 1e7)
        direction = SkyCoord('00h42m30s', '-41d12m00s', frame='icrs')
        vt = Visibility()
        vt = visibility_from_configuration(config, times, freq, weight=1.0, direction=direction)
        print(vt.data)
        print(vt.frequency)
        assert len(numpy.unique(vt.data['time'])) == len(times)

if __name__ == '__main__':
    unittest.main()
