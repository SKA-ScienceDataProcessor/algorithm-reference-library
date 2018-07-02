""" Unit tests for image operations


"""
import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from processing_components.convolution_function.kernels import create_awterm_convolutionfunction, \
    create_pswf_convolutionfunction, create_box_convolutionfunction
from processing_components.griddata.gridding import grid_visibility_to_griddata, \
    grid_visibility_to_griddata_fast
from processing_components.griddata.operations import create_griddata_from_image
from processing_components.image.operations import export_image_to_fits
from processing_components.image.operations import smooth_image
from processing_components.imaging.base import normalize_sumwt
from processing_components.imaging.base import predict_skycomponent_visibility
from processing_components.imaging.primary_beams import create_pb_generic
from processing_components.simulation.testing_support import create_named_configuration, create_unittest_model, \
    create_unittest_components, ingest_unittest_visibility
from processing_components.skycomponent.operations import insert_skycomponent

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))

log = logging.getLogger(__name__)


class TestGridDataGridding(unittest.TestCase):
    
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
    
    def actualSetUp(self):
        self.npixel = 512
        self.low = create_named_configuration('LOWBD2', rmax=300.0)
        self.freqwin = 5
        self.vis_list = list()
        self.ntimes = 3
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        self.channelwidth = numpy.array(self.freqwin * [self.frequency[1] - self.frequency[0]])
        
        self.vis_pol = PolarisationFrame('linear')
        self.image_pol = PolarisationFrame('stokesIQUV')
        
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        
        flux = numpy.array([f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = ingest_unittest_visibility(self.low,
                                              self.frequency,
                                              self.channelwidth,
                                              self.times,
                                              self.vis_pol,
                                              self.phasecentre,
                                              block=False,
                                              zerow=True)
        
        self.model = create_unittest_model(self.vis, self.image_pol, cellsize=0.0015,
                                           npixel=self.npixel, nchan=self.freqwin)
        self.components = create_unittest_components(self.model, flux, applypb=False)
        self.model = insert_skycomponent(self.model, self.components)
        self.vis = predict_skycomponent_visibility(self.vis, self.components)
        
        # Calculate the model convolved with a Gaussian.
        self.cmodel = smooth_image(self.model)
        export_image_to_fits(self.model, '%s/test_gridding_model.fits' % self.dir)
        export_image_to_fits(self.cmodel, '%s/test_gridding_cmodel.fits' % self.dir)
    
    def test_time_setup(self):
        self.actualSetUp()
    
    def test_convolution_mapping_pswf(self):
        self.actualSetUp()
        gcf, cf = create_pswf_convolutionfunction(self.model, support=6, oversampling=16)
        griddata = create_griddata_from_image(self.model)
        im, sumwt = grid_visibility_to_griddata(self.vis, griddata=griddata, gcf=gcf, cf=cf)
        im = normalize_sumwt(im, sumwt)
        export_image_to_fits(im, '%s/test_gridding_dirty_pswf.fits' % self.dir)
    
    def test_convolution_mapping_aterm(self):
        self.actualSetUp()
        pb = create_pb_generic(self.model, diameter=35.0, blockage=0.0)
        export_image_to_fits(pb, "%s/test_convolutionfunction_aterm_pb.fits" % self.dir)
        gcf, cf = create_awterm_convolutionfunction(self.model, pb=pb, nw=1, wstep=0.0, use_aaf=True,
                                                    oversampling=16, support=16)
        griddata = create_griddata_from_image(self.model)
        im, sumwt = grid_visibility_to_griddata(self.vis, griddata=griddata, gcf=gcf, cf=cf)
        im = normalize_sumwt(im, sumwt)
        export_image_to_fits(im, '%s/test_gridding_dirty_aterm.fits' % self.dir)
    
    def test_convolution_mapping_pswf_nooversampling(self):
        self.actualSetUp()
        gcf, cf = create_pswf_convolutionfunction(self.model, support=6, oversampling=1)
        griddata = create_griddata_from_image(self.model)
        im, sumwt = grid_visibility_to_griddata(self.vis, griddata=griddata, gcf=gcf, cf=cf)
        im = normalize_sumwt(im, sumwt)
        export_image_to_fits(im, '%s/test_gridding_dirty_pswf_nooversampling.fits' % self.dir)
    
    def test_convolution_mapping_box(self):
        self.actualSetUp()
        gcf, cf = create_box_convolutionfunction(self.model)
        griddata = create_griddata_from_image(self.model)
        im, sumwt = grid_visibility_to_griddata(self.vis, griddata=griddata, gcf=gcf, cf=cf)
        im = normalize_sumwt(im, sumwt)
        export_image_to_fits(im, '%s/test_gridding_dirty_box.fits' % self.dir)
    
    def test_convolution_mapping_fast(self):
        self.actualSetUp()
        gcf, cf = create_box_convolutionfunction(self.model)
        griddata = create_griddata_from_image(self.model)
        im, sumwt = grid_visibility_to_griddata_fast(self.vis, griddata=griddata, gcf=gcf, cf=cf)
        im = normalize_sumwt(im, sumwt)
        export_image_to_fits(im, '%s/test_gridding_dirty_fast.fits' % self.dir)


if __name__ == '__main__':
    unittest.main()
