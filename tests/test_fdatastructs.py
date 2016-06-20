import unittest

import numpy
from numpy.testing import assert_allclose

from functions.fcomp import fcomp_construct
from functions.fconfig import fconfig_from_name
from functions.fgaintab import fgaintab_from_array
from functions.fimage import fimage, fimage_add, fimage_from_array, fimage_from_fits
from functions.fskymod import fskymod, fskymod_from_fimage, fskymod_add_fcomp
from functions.fvistab import fvistab, fvistab_from_fconfig

from astropy.coordinates import SkyCoord


class TestFunctions(unittest.TestCase):
    def test_fcomponent(self):
        flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
        comp = fcomp_construct(direction, flux, shape='Point')

    def test_fconfiguration(self):
        for telescope in ['LOWBD1', 'LOWBD2', 'LOFAR', 'VLAA']:
            fc = fconfig_from_name(telescope)
            print(fc.location.to_geodetic())

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
        print(fgaintab_from_array(gains, times, antennas, weight).data)

    def test_fimage(self):
        m31model = fimage_from_fits("../data/models/m31.model.fits")
        m31max = m31model.data.max()
        m31model_by_array = fimage()
        m31model_by_array = fimage_from_array(m31model.data, m31model.wcs)
        m31model = fimage_add(m31model, m31model_by_array)
        assert_allclose(m31model.data.max(), 2.0 * m31max, atol=1e-15)

    def test_skymodel(self):
        m31image = fimage_from_fits("../data/models/m31.model.fits")
        m31sm = fskymod_from_fimage(m31image)
        direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
        flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        comp = fcomp_construct(direction, flux, shape='Point')
        m31sm=fskymod_add_fcomp(m31sm, comp)

    def test_fvistable(self):
        config = fconfig_from_name('VLAA')
        times = numpy.arange(-3.0, +3.0, 3.0 / 60.0) * numpy.pi / 12.0
        freq = numpy.arange(5e6, 150.0e6, 1e7)
        direction = SkyCoord('00h42m30s', '-41d12m00s', frame='icrs')
        vt = fvistab()
        vt = fvistab_from_fconfig(config, times, freq, weight=1.0, direction=direction)
        print(vt.data)
        print(vt.frequency)
        assert len(numpy.unique(vt.data['time'])) == len(times)


if __name__ == '__main__':
    unittest.main()
