import unittest

import numpy
from numpy.testing import assert_allclose

from arl.define_skymodel import SkyComponent, create_skycomponent
from arl.simulate_visibility import create_named_configuration, create_gaintable_from_array
from arl.define_image import add_image, create_image_from_array, create_image_from_fits
from arl.define_skymodel import create_skymodel_from_image, add_component_to_skymodel
from arl.define_visibility import Visibility, create_visibility

from astropy.coordinates import SkyCoord


class TestDataStructures(unittest.TestCase):

    def test_component(self):
        flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
        frequency=numpy.arange(1.0e8,1.5e8,3e7)
        comp = create_skycomponent(flux, flux=frequency, frequency=frequency, shape='Point')

    def test_configuration(self):
        for telescope in ['LOWBD1', 'LOWBD2', 'LOFAR', 'VLAA']:
            fc = create_named_configuration(telescope)
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
        print(create_gaintable_from_array(gain=gain, time=times, antenna=antennas, weight=weight,
                                          frequency=frequency).data)

    def test_image(self):
        m31model = create_image_from_fits("./data/models/M31.MOD")
        m31max = m31model.data.max()
        m31model_by_array = create_image_from_array(m31model.data, m31model.wcs)
        m31model = add_image(m31model, m31model_by_array)
        assert_allclose(m31model.data.max(), 2.0 * m31max, atol=1e-15)

    def test_skymodel(self):
        m31image = create_image_from_fits("./data/models/M31.MOD")
        m31sm = create_skymodel_from_image(m31image)
        direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
        flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        frequency=numpy.arange(1.0e8,1.5e8,1.0e7)
        comp = create_skycomponent(flux=flux, direction=direction, frequency=frequency, shape='Point')
        m31sm = add_component_to_skymodel(m31sm, comp)

    def test_visibility(self):
        config = create_named_configuration('VLAA')
        times = numpy.arange(-3.0, +3.0, 3.0 / 60.0) * numpy.pi / 12.0
        freq = numpy.arange(5e6, 150.0e6, 1e7)
        direction = SkyCoord('00h42m30s', '-41d12m00s', frame='icrs')
        vt = Visibility()
        vt = create_visibility(config, times, freq, weight=1.0, phasecentre=direction)
        print(vt.data)
        print(vt.frequency)
        assert len(numpy.unique(vt.data['time'])) == len(times)


if __name__ == '__main__':
    unittest.main()
