import logging
import unittest

import numpy
from astropy.coordinates import SkyCoord

from arl.data.parameters import crocodile_path
from arl.image.image_operations import add_image, create_image_from_array
from arl.skymodel.skymodel_operations import create_skycomponent
from arl.skymodel.skymodel_operations import create_skymodel_from_image, add_component_to_skymodel
from arl.util.testing_support import create_test_image, create_named_configuration, import_visibility_from_oskar
from arl.visibility.visibility_calibration import create_gaintable_from_array
from arl.visibility.visibility_operations import create_visibility

log = logging.getLogger("tests.TestDataModels")

class TestDataModels(unittest.TestCase):

    def test_component(self):
        flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
        frequency=numpy.arange(1.0e8,1.5e8,3e7)
        comp = create_skycomponent(direction=direction, flux=flux, frequency=frequency, shape='Point')

    def test_configuration(self):
        for telescope in ['LOWBD1', 'LOWBD2', 'LOFAR', 'VLAA']:
            fc = create_named_configuration(telescope)
            log.debug(fc.location.to_geodetic())

    def test_gaintable(self):
        nant = 27
        npol = 4
        atimes = numpy.arange(0.0, 43200.0, 10.0)
        ntimes = len(atimes)
        times = numpy.repeat(atimes, nant)
        antennas = numpy.array(ntimes * list(range(nant)))
        frequency=numpy.arange(1.0e8,1.5e8,1.0e7)
        gain = numpy.ones([len(times), len(frequency), npol], dtype='complex')
        weight = numpy.ones([len(times), len(frequency)], dtype='float')
        gaintab = create_gaintable_from_array(gain=gain, time=times, antenna=antennas, weight=weight,
                                              frequency=frequency)
        log.debug(gaintab.data)

    def test_image(self):
        m31model = create_test_image()
        m31max = m31model.data.max()
        m31model_by_array = create_image_from_array(m31model.data, m31model.wcs)
        m31model = add_image(m31model, m31model_by_array)
        self.assertAlmostEqual(m31model.data.max(), 2.0 * m31max, places=7)

    def test_skymodel(self):
        m31image = create_test_image()
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
        vis = create_visibility(config, times, freq, weight=1.0, phasecentre=direction)
        log.debug(vis.data)
        log.debug(vis.frequency)
        self.assertEqual(len(numpy.unique(vis.data['time'])), len(times))

    def test_visibility_from_oskar(self):
        for oskar_file in ["data/vis/vla_1src_6h/test_vla.vis",
                           "data/vis/vla_grid_6h/test_vla.vis"]:
            vis = import_visibility_from_oskar(crocodile_path(oskar_file))
            self.assertEqual(len(numpy.unique(vis.antenna1))+1, len(vis.configuration.xyz))
            self.assertEqual(len(numpy.unique(vis.antenna2))+1, len(vis.configuration.xyz))

if __name__ == '__main__':
    unittest.main()
