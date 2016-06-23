import unittest

import numpy
from numpy.testing import assert_allclose

from functions.skycomponent import SkyComponent
from functions.configuration import named_configuration
from functions.gaintable import gaintable_from_array
from functions.image import image_add, image_from_array, image_from_fits
from functions.skymodel import skymodel_from_image, skymodel_add_component
from functions.visibility import Visibility, simulate

from astropy.coordinates import SkyCoord


class TestFunctions(unittest.TestCase):

    def test_component(self):
        flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
        comp = SkyComponent(direction, flux, shape='Point')

    def test_configuration(self):
        for telescope in ['LOWBD1', 'LOWBD2', 'LOFAR', 'VLAA']:
            fc = named_configuration(telescope)
            print(fc.location.to_geodetic())

    def test_gaintable(self):
        nant = 27
        aantennas = numpy.arange(nant, dtype='int')
        npol = 4
        atimes = numpy.arange(0.0, 43200.0, 10.0)
        ntimes = len(atimes)
        times = numpy.repeat(atimes, nant)
        antennas = numpy.array(ntimes * list(range(nant)))
        frequency=numpy.arange(1.0e8,1.5e8,1.0e7)
        nrows = len(times)
        gain = numpy.ones([len(times), len(frequency), npol], dtype='complex')
        weight = numpy.ones([len(times), len(frequency)], dtype='float')
        print(gaintable_from_array(gain=gain, time=times, antenna=antennas, weight=weight,
                                   frequency=frequency).data)

    def test_image(self):
        m31model = image_from_fits("./data/models/M31.MOD")
        m31max = m31model.data.max()
        m31model_by_array = image_from_array(m31model.data, m31model.wcs)
        m31model = image_add(m31model, m31model_by_array)
        assert_allclose(m31model.data.max(), 2.0 * m31max, atol=1e-15)

    def test_skymodel(self):
        m31image = image_from_fits("./data/models/M31.MOD")
        m31sm = skymodel_from_image(m31image)
        direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
        flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        comp = SkyComponent(direction, flux, shape='Point')
        m31sm = skymodel_add_component(m31sm, comp)

    def test_visibility(self):
        config = named_configuration('VLAA')
        times = numpy.arange(-3.0, +3.0, 3.0 / 60.0) * numpy.pi / 12.0
        freq = numpy.arange(5e6, 150.0e6, 1e7)
        direction = SkyCoord('00h42m30s', '-41d12m00s', frame='icrs')
        vt = Visibility()
        vt = simulate(config, times, freq, weight=1.0, direction=direction)
        print(vt.data)
        print(vt.frequency)
        assert len(numpy.unique(vt.data['time'])) == len(times)


if __name__ == '__main__':
    unittest.main()
