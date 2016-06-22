import unittest

import numpy
from numpy.testing import assert_allclose

from functions.component import component
from functions.configuration import named_configuration
from functions.gaintable import gaintable_from_array
from functions.image import image, image_add, image_from_array, image_from_fits
from functions.skymodel import skymodel, skymodel_from_image, skymodel_add_component
from functions.visibility import visibility, visibility_from_configuration

from astropy.coordinates import SkyCoord


class TestFunctions(unittest.TestCase):
    def test_componentonent(self):
        flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
        comp = component_construct(direction, flux, shape='Point')

    def test_configurationuration(self):
        for telescope in ['LOWBD1', 'LOWBD2', 'LOFAR', 'VLAA']:
            fc = named_configuration(telescope)
            print(fc.location.to_geodetic())

    def test_gaintablele(self):
        nant = 27
        aantennas = numpy.arange(nant, dtype='int')
        npol = 4
        freq = numpy.arange(5.e7, 15.e7, 2.5e7)
        print(freq)
        atimes = numpy.arange(0.0, 43200.0, 10.0)
        ntimes = len(atimes)
        times = numpy.repeat(atimes, nant)
        antennas = numpy.array(ntimes * list(range(nant)))
        nrows = len(times)
        gains = numpy.ones([len(times), len(freq), npol], dtype='complex')
        weight = numpy.ones([len(times), len(freq)], dtype='float')
        print(gaintable_from_array(gains, times, antennas, weight).data)

    def test_image(self):
        m31model = image_from_fits("./data/models/m31.model.fits")
        m31max = m31model.data.max()
        m31model_by_array = image()
        m31model_by_array = image_from_array(m31model.data, m31model.wcs)
        m31model = image_add(m31model, m31model_by_array)
        assert_allclose(m31model.data.max(), 2.0 * m31max, atol=1e-15)

    def test_skymodel(self):
        m31image = image_from_fits("./data/models/m31.model.fits")
        m31sm = skymodel_from_image(m31image)
        direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
        flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        comp = component(direction, flux, shape='Point')
        m31sm=skymodel_add_component(m31sm, comp)

    def test_visibilityle(self):
        config = named_configuration('VLAA')
        times = numpy.arange(-3.0, +3.0, 3.0 / 60.0) * numpy.pi / 12.0
        freq = numpy.arange(5e6, 150.0e6, 1e7)
        direction = SkyCoord('00h42m30s', '-41d12m00s', frame='icrs')
        vt = visibility()
        vt = visibility_from_configuration(config, times, freq, weight=1.0, direction=direction)
        print(vt.data)
        print(vt.frequency)
        assert len(numpy.unique(vt.data['time'])) == len(times)

if __name__ == '__main__':
    unittest.main()
