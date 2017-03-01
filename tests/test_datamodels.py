import logging
import unittest

import numpy
from astropy.coordinates import SkyCoord

from arl.data.data_models import *
from arl.data.parameters import arl_path
from arl.image.operations import add_image, create_image_from_array
from arl.skymodel.operations import create_skycomponent
from arl.skymodel.operations import create_skymodel_from_image
from arl.util.testing_support import create_test_image, create_named_configuration
from arl.visibility.operations import create_visibility

log = logging.getLogger(__name__)

class TestDataModels(unittest.TestCase):

    def test_component(self):
        flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        direction = SkyCoord('00h42m30s', '-41d12m00s', frame='icrs')
        frequency=numpy.arange(1.0e8,1.5e8,3e7)
        comp = create_skycomponent(direction=direction, flux=flux, frequency=frequency, shape='Point')
        assert comp.flux.shape == (2,4)
        assert comp.direction.separation(direction) == 0.0
        assert comp.shape == 'Point'

    def test_configuration(self):
        for telescope in ['LOWBD1', 'LOWBD2', 'LOWBD2-CORE', 'LOFAR', 'VLAA', 'VLAA_north']:
            fc = create_named_configuration(telescope)
            log.debug(fc.location.to_geodetic())
            log.debug(fc.size())


    def test_image(self):
        m31model = create_test_image()
        m31max = m31model.data.max()
        m31model_by_array = create_image_from_array(m31model.data, m31model.wcs)
        m31model = add_image(m31model, m31model_by_array, docheckwcs=True)
        self.assertAlmostEqual(m31model.data.max(), 2.0 * m31max, places=7)

    def test_skymodel(self):
        m31image = create_test_image()
        m31sm = create_skymodel_from_image(m31image)
        direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
        flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        frequency=numpy.arange(1.0e8,1.5e8,2.5e7)
        comp = create_skycomponent(flux=flux, direction=direction, frequency=frequency, shape='Point')
        log.debug("%d %d" % (comp.nchan, comp.npol))

    def test_visibility(self):
        config = create_named_configuration('VLAA')
        times = numpy.arange(-3.0, +3.0, 3.0 / 60.0) * numpy.pi / 12.0
        freq = numpy.arange(5e6, 150.0e6, 1e7)
        direction = SkyCoord('00h42m30s', '-41d12m00s', frame='icrs')
        vis = create_visibility(config, times, freq, phasecentre=direction, weight=1.0)
        log.debug(vis.data)
        log.debug(vis.frequency)
        self.assertEqual(len(numpy.unique(vis.data['time'])), len(times))

if __name__ == '__main__':
    unittest.main()
