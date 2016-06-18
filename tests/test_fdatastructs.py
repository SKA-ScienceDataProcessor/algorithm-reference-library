import unittest

import numpy
from numpy.testing import assert_allclose

from functions.fcomponent import fcomponent
from functions.fconfiguration import fconfiguration
from functions.fgaintable import fgaintable
from functions.fimage import fimage
from functions.fskymodel import fskymodel
from functions.fvistable import fvistable
from astropy.coordinates import SkyCoord


class TestFunctions(unittest.TestCase):
    def test_fcomponent(self):
        flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
        comp = fcomponent(direction, flux, shape='Point')

    def test_fconfiguration(self):
        fc = fconfiguration()
        for telescope in fc.known():
            config = fconfiguration()
            config.fromname(telescope)
            geodetic = config.location.to_geodetic()

    def test_fgaintable(self):
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
        gt = fgaintable(gains, times, antennas, weight)

    def test_fimage(self):
        m31model = fimage()
        m31model.from_fits("../data/models/m31.model.fits")
        m31max = m31model.data.max()
        m31model_by_array = fimage()
        m31model_by_array.from_array(m31model.data, m31model.wcs)
        m31model += m31model_by_array
        assert_allclose(m31model.data.max(), 2.0 * m31max, atol=1e-15)

    def test_skymodel(self):
        m31image = fimage().from_fits("../data/models/m31.model.fits")
        m31sm = fskymodel(m31image)
        direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
        flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        comp = fcomponent(direction, flux, shape='Point')
        m31sm.addcomponents(comp)

    def test_fvistable(self):
        config = fconfiguration().fromname('VLAA')
        times = numpy.arange(-3.0, +3.0, 3.0 / 60.0) * numpy.pi / 12.0
        freq = numpy.arange(5e6, 150.0e6, 1e7)
        direction = SkyCoord('00h42m30s', '-41d12m00s', frame='icrs')
        vt = fvistable()
        vt.observe(config, times, freq, weight=1.0, direction=direction)
        print(vt)
        print(vt.freq)
        assert len(numpy.unique(vt['time'])) == len(times)


if __name__ == '__main__':
    unittest.main()
