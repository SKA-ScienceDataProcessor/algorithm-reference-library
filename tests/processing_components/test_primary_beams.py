"""Unit tests for testing support


"""

import logging
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.data_models.polarisation import PolarisationFrame

from arl.processing_components.image.operations import export_image_to_fits
from arl.processing_components.imaging.base import create_image_from_visibility
from arl.processing_components.imaging.primary_beams import create_pb, create_vp
from arl.processing_components.simulation import create_named_configuration
from arl.processing_components.visibility.base import create_visibility

log = logging.getLogger(__name__)


class TestPrimaryBeams(unittest.TestCase):
    def setUp(self):
        from arl.data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.persist = False
        
    def createVis(self, config='MID', dec=-35.0, rmax=1e3, freq=1.3e9):
        self.frequency = [freq]
        self.channel_bandwidth = [1e6]
        self.flux = numpy.array([[100.0]])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.config = create_named_configuration(config)
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        nants = self.config.xyz.shape[0]
        self.npixel = 1024
        self.fov = 8
        self.cellsize = numpy.pi * self.fov / (self.npixel * 180.0)
        assert nants > 1
        assert len(self.config.names) == nants
        assert len(self.config.mount) == nants
    
        self.config = create_named_configuration(config, rmax=rmax)
        self.phasecentre = SkyCoord(ra=+15 * u.deg, dec=dec * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_visibility(self.config, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('stokesI'))

    def test_create_primary_beams_RADEC(self):
        self.createVis()
        for telescope in ['VLA', 'ASKAP', 'MID', 'LOW']:
            model = create_image_from_visibility(self.vis, cellsize=self.cellsize, npixel=self.npixel, override_cellsize=False)
            beam = create_pb(model, telescope=telescope, use_local=False)
            assert numpy.max(beam.data) > 0.0
            if self.persist: export_image_to_fits(beam, "%s/test_primary_beam_RADEC_%s.fits" % (self.dir, telescope))

    def test_create_primary_beams_AZELGEO(self):
        self.createVis()
        for telescope in ['VLA', 'ASKAP', 'MID', 'MID_GAUSS', 'MID_FEKO_B1', 'MID_FEKO_B2', 'MID_FEKO_Ku', 'MID_GRASP', 'LOW']:
            model = create_image_from_visibility(self.vis, cellsize=self.cellsize, npixel=self.npixel, override_cellsize=False)
            beam = create_pb(model, telescope=telescope, use_local=True)
            assert numpy.max(beam.data) > 0.0
            if self.persist: export_image_to_fits(beam, "%s/test_primary_beam_AZELGEO_%s.fits" % (self.dir, telescope))

    def test_create_voltage_patterns(self):
        self.createVis()
        for telescope in ['VLA', 'ASKAP', 'LOW']:
            model = create_image_from_visibility(self.vis, cellsize=self.cellsize, npixel=self.npixel, override_cellsize=False)
            beam=create_vp(model, telescope=telescope)
            assert numpy.max(numpy.abs(beam.data.real)) > 0.0
            assert numpy.max(numpy.abs(beam.data.imag)) < 1e-15, numpy.max(numpy.abs(beam.data.imag))

    def test_create_voltage_patterns_MID_GAUSS(self):
        self.createVis()
        model = create_image_from_visibility(self.vis, npixel=self.npixel, cellsize=self.cellsize,
                                             override_cellsize=False)
        for telescope in ['MID_GAUSS']:
            beam=create_vp(model, telescope=telescope, padding=4)
            beam_data = beam.data
            beam.data = numpy.real(beam_data)
            if self.persist: export_image_to_fits(beam, "%s/test_voltage_pattern_real_%s.fits" % (self.dir, telescope))
            beam.data = numpy.imag(beam_data)
            if self.persist: export_image_to_fits(beam, "%s/test_voltage_pattern_imag_%s.fits" % (self.dir, telescope))

    def test_create_voltage_patterns_MID(self):
        self.createVis(freq=1.4e9)
        model = create_image_from_visibility(self.vis, npixel=self.npixel, cellsize=self.cellsize,
                                             override_cellsize=False)
        for telescope in ['MID', 'MID_FEKO_B1', 'MID_FEKO_B2', 'MID_FEKO_Ku']:
            beam=create_vp(model, telescope=telescope, padding=4)
            beam_data = beam.data
            beam.data = numpy.real(beam_data)
            beam.wcs.wcs.crval[0]=0.0
            beam.wcs.wcs.crval[1]=90.0
            if self.persist: export_image_to_fits(beam, "%s/test_voltage_pattern_real_zenith_%s.fits" % (self.dir, telescope))


if __name__ == '__main__':
    unittest.main()
