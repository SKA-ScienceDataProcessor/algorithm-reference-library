""" Unit tests for pipelines expressed via dask.bag


"""

import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord
from dask import bag

from arl.calibration.operations import create_gaintable_from_blockvisibility, apply_gaintable
from arl.data.polarisation import PolarisationFrame
from arl.graphs.bags import reify
from arl.util.bag_support import gleam_model_bag
from arl.image.operations import qa_image, export_image_to_fits, copy_image, \
    create_empty_image_like
from arl.imaging import create_image_from_visibility, predict_skycomponent_visibility, \
    predict_skycomponent_blockvisibility
from arl.skycomponent.operations import create_skycomponent, insert_skycomponent
from arl.util.testing_support import create_named_configuration
from arl.util.testing_support import simulate_gaintable
from arl.visibility.base import create_visibility, create_blockvisibility


class TestDaskBagsSupport(unittest.TestCase):
    
    def setUp(self):
        
        # We can compute using the default scheduler. Using the distributed scheduler within
        # jenkins does not work.
        self.compute = True
        
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        
        self.npixel = 512
        
        self.setupVis(add_errors=False, block=False)
    
    def setupVis(self, add_errors=False, block=True, freqwin=3):
        self.freqwin = freqwin
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        self.vis_bag = \
            bag.from_sequence([{'freqwin': f,
                                'vis': self.ingest_visibility([freq], times=self.times,
                                                              add_errors=add_errors,
                                                              block=block)}
                               for f, freq in enumerate(self.frequency)])
        
        self.vis_bag = reify(self.vis_bag)
        self.model_bag = bag.from_sequence(self.freqwin * [self.model])
        self.empty_model_bag = bag.from_sequence(self.freqwin * [self.empty_model])
    
    def ingest_visibility(self, freq=[1e8], chan_width=[1e6], times=None, reffrequency=None, add_errors=False,
                          block=True):
        if times is None:
            times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 5)

        if reffrequency is None:
            reffrequency = [1e8]
        lowcore = create_named_configuration('LOWBD2-CORE')
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
                                             frequency=reffrequency, phasecentre=phasecentre,
                                             polarisation_frame=PolarisationFrame("stokesI"))
        flux = numpy.array([[100.0]])
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
                comp = create_skycomponent(flux=flux, frequency=frequency, direction=sc,
                                           polarisation_frame=PolarisationFrame("stokesI"))
                comps.append(comp)
        if block:
            predict_skycomponent_blockvisibility(vt, comps)
        else:
            predict_skycomponent_visibility(vt, comps)
        insert_skycomponent(model, comps)
        self.model = copy_image(model)
        self.empty_model = create_empty_image_like(model)
        
        export_image_to_fits(model, '%s/test_bags_model.fits' % (self.dir))
        
        if add_errors:
            # These will be the same for all calls
            numpy.random.seed(180555)
            gt = create_gaintable_from_blockvisibility(vt)
            gt = simulate_gaintable(gt, phase_error=1.0, amplitude_error=0.0)
            vt = apply_gaintable(vt, gt)
        return vt
    

    def test_gleam_model_bag(self):
        gmb = gleam_model_bag(npixel=256)
        qa = qa_image(gmb.compute()[0]['image'])
        assert abs(qa.data['max'] - 5.00949445505) < 1e-7, qa
        assert abs(qa.data['sum'] - 138.84405529) < 1e-7, qa
