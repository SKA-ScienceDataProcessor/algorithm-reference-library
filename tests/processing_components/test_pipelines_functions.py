""" Unit tests for pipelines


"""

import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord

from data_models.polarisation import PolarisationFrame

from processing_components.calibration.operations import qa_gaintable, create_gaintable_from_blockvisibility, apply_gaintable
from processing_components.image.operations import copy_image, create_empty_image_like, export_image_to_fits
from processing_components.imaging.base import predict_skycomponent_visibility, \
    create_image_from_visibility
from processing_components.skycomponent.operations import create_skycomponent, insert_skycomponent
from processing_components.simulation.testing_support import create_named_configuration, simulate_gaintable
from processing_components.visibility.base import create_blockvisibility, create_visibility

from processing_components.functions.pipeline_functions import rcal

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestPipelinesFunctions(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.setupVis(add_errors=False, block=True)
    
    def setupVis(self, add_errors=False, block=True, freqwin=7, bandpass=False):
        self.npixel = 256
   
        self.freqwin = freqwin
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        if freqwin > 1:
            self.channel_bandwidth = numpy.array(freqwin * [self.frequency[1] - self.frequency[0]])
        else:
            self.channel_bandwidth = numpy.array([4e7])
        self.vis = self.ingest_visibility(self.frequency, chan_width=self.channel_bandwidth,
                                          times=self.times, add_errors=add_errors, block=block,
                                          bandpass=bandpass)
    
    def ingest_visibility(self, freq=None, chan_width=None, times=None, add_errors=False,
                          block=True, bandpass=False):
        if freq is None:
            freq = [1e8]
        if chan_width is None:
            chan_width = [1e6]
        if times is None:
            times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 5)
        
        lowcore = create_named_configuration('LOWBD2', rmax=750.0)
        frequency = numpy.array(freq)
        channel_bandwidth = numpy.array(chan_width)
        
        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        if block:
            vt = create_blockvisibility(lowcore, times, frequency, channel_bandwidth=channel_bandwidth,
                                        weight=1.0, phasecentre=phasecentre,
                                        polarisation_frame=PolarisationFrame("stokesI"))
        else:
            vt = create_visibility(lowcore, times, frequency, channel_bandwidth=channel_bandwidth,
                                   weight=1.0, phasecentre=phasecentre,
                                   polarisation_frame=PolarisationFrame("stokesI"))
        cellsize = 0.001
        model = create_image_from_visibility(vt, npixel=self.npixel, cellsize=cellsize, npol=1,
                                             frequency=frequency, phasecentre=phasecentre,
                                             polarisation_frame=PolarisationFrame("stokesI"))
        nchan = len(self.frequency)
        flux = numpy.array(nchan * [[100.0]])
        facets = 4
        
        rpix = model.wcs.wcs.crpix - 1.0
        spacing_pixels = self.npixel // facets
        centers = [-1.5, -0.5, 0.5, 1.5]
        comps = list()
        for iy in centers:
            for ix in centers:
                p = int(round(rpix[0] + ix * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[0]))), \
                    int(round(rpix[1] + iy * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[1])))
                sc = pixel_to_skycoord(p[0], p[1], model.wcs, origin=1)
                comp = create_skycomponent(direction=sc, flux=flux, frequency=frequency,
                                           polarisation_frame=PolarisationFrame("stokesI"))
                comps.append(comp)
        if block:
            predict_skycomponent_visibility(vt, comps)
        else:
            predict_skycomponent_visibility(vt, comps)
        insert_skycomponent(model, comps)
        self.comps = comps
        self.model = copy_image(model)
        self.empty_model = create_empty_image_like(model)
        export_image_to_fits(model, '%s/test_pipeline_functions_model.fits' % (self.dir))
        
        if add_errors:
            # These will be the same for all calls
            numpy.random.seed(180555)
            gt = create_gaintable_from_blockvisibility(vt)
            gt = simulate_gaintable(gt, phase_error=1.0, amplitude_error=0.0)
            vt = apply_gaintable(vt, gt)
            
            if bandpass:
                bgt = create_gaintable_from_blockvisibility(vt, timeslice=1e5)
                bgt = simulate_gaintable(bgt, phase_error=0.01, amplitude_error=0.01, smooth_channels=4)
                vt = apply_gaintable(vt, bgt)
        
        return vt

    def test_time_setup(self):
        pass
        
    def test_RCAL(self):
        self.setupVis(add_errors=True, block=True, freqwin=5)
        for igt, gt in enumerate(rcal(vis=self.vis, components=self.comps)):
            log.info("Chunk %d: %s" % (igt, qa_gaintable(gt)))


if __name__ == '__main__':
    unittest.main()
