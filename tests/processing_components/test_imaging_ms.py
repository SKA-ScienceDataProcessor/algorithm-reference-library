""" Unit tests for pipelines expressed via dask.delayed


"""
import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from processing_components.imaging.base import predict_skycomponent_visibility
from processing_components.simulation.configurations import create_named_configuration
from processing_components.simulation.testing_support import ingest_unittest_visibility, \
    create_unittest_model, create_unittest_components
from processing_components.skycomponent.operations import insert_skycomponent
from processing_components.visibility.coalesce import convert_blockvisibility_to_visibility

try:
    import casacore
    from casacore.tables import table  # pylint: disable=import-error
    from processing_components.visibility.base import create_blockvisibility, create_blockvisibility_from_ms
    from processing_components.visibility.base import export_blockvisibility_to_ms
    
    run_ms_tests = True
#            except ModuleNotFoundError:
except:
    run_ms_tests = False

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImaging(unittest.TestCase):
    def setUp(self):
        
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
    
    def actualSetUp(self, freqwin=1, block=True, dopol=False):
        
        self.npixel = 512
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = freqwin
        self.vis = list()
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0

        if dopol:
            self.vis_pol = PolarisationFrame('linear')
            self.image_pol = PolarisationFrame('stokesIQUV')
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        else:
            self.vis_pol = PolarisationFrame('stokesI')
            self.image_pol = PolarisationFrame('stokesI')
            f = numpy.array([100.0])

        if freqwin > 1:
            self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
            self.channelwidth = numpy.array(freqwin * [self.frequency[1] - self.frequency[0]])
            flux = numpy.array([f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency])
        else:
            self.frequency = numpy.array([1e8])
            self.channelwidth = numpy.array([1e6])
            flux = numpy.array([f])

        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.bvis = ingest_unittest_visibility(self.low,
                                               self.frequency,
                                               self.channelwidth,
                                               self.times,
                                               self.vis_pol,
                                               self.phasecentre, block=block)
        
        self.vis = convert_blockvisibility_to_visibility(self.bvis)
        
        self.model = create_unittest_model(self.vis, self.image_pol, npixel=self.npixel, nchan=freqwin)
        
        self.components = create_unittest_components(self.model, flux)
        
        self.model = insert_skycomponent(self.model, self.components)
        
        self.bvis = predict_skycomponent_visibility(self.bvis, self.components)
    
    
    @unittest.skipUnless(run_ms_tests, "requires the 'casacore' module")
    def test_export_ms(self):
        self.actualSetUp()
        msoutfile = "%s/test_imaging_ms_%dfreqwin.ms" % (self.dir, len(self.frequency))
        export_blockvisibility_to_ms(msoutfile, [self.bvis], source_name='M31')

    @unittest.skipUnless(run_ms_tests, "requires the 'casacore' module")
    def test_export_ms_7freqwin(self):
        self.actualSetUp(freqwin=7)
        msoutfile = "%s/test_imaging_ms_%dfreqwin.ms" % (self.dir, len(self.frequency))
        export_blockvisibility_to_ms(msoutfile, [self.bvis], source_name='M31')

    @unittest.skipUnless(run_ms_tests, "requires the 'casacore' module")
    def test_export_ms_pol(self):
        self.actualSetUp(dopol=True)
        msoutfile = "%s/test_imaging_ms_pol_%dfreqwin.ms" % (self.dir, len(self.frequency))
        export_blockvisibility_to_ms(msoutfile, [self.bvis], source_name='M31')

    @unittest.skipUnless(run_ms_tests, "requires the 'casacore' module")
    def test_export_ms_7freqwin_pol(self):
        self.actualSetUp(freqwin=7, dopol=True)
        msoutfile = "%s/test_imaging_ms_pol_%dfreqwin.ms" % (self.dir, len(self.frequency))
        export_blockvisibility_to_ms(msoutfile, [self.bvis], source_name='M31')



if __name__ == '__main__':
    unittest.main()
