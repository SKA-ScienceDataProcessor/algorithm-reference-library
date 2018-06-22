""" Unit tests for image operations


"""
import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from processing_components.griddata.gridding import convolution_mapping, grid_visibility_to_griddata
from processing_components.griddata.kernels import create_wterm_kernel, create_pswf_kernel, \
    convert_griddata_to_convfunction
from processing_components.griddata.operations import create_griddata_from_image, convert_griddata_to_image
from processing_components.image.operations import export_image_to_fits
from processing_components.image.operations import smooth_image
from processing_components.imaging.base import predict_skycomponent_visibility
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
        
        self.npixel = 256
        self.low = create_named_configuration('LOWBD2', rmax=300.0)
        self.freqwin = 5
        self.vis_list = list()
        self.ntimes = 2
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
                                              block=False)
        
        self.model = create_unittest_model(self.vis, self.image_pol, cellsize=0.001,
                                           npixel=self.npixel, nchan=self.freqwin)
        self.components = create_unittest_components(self.model, flux)
        self.model = insert_skycomponent(self.model, self.components)
#        self.vis = predict_skycomponent_visibility(self.vis, self.components)
        
        # Calculate the model convolved with a Gaussian.
        self.cmodel = smooth_image(self.model)
        export_image_to_fits(self.model, '%s/test_gridding_model.fits' % self.dir)
        export_image_to_fits(self.cmodel, '%s/test_gridding_cmodel.fits' % self.dir)
        
    def test_time_setup(self):
            
            self.actualSetUp()
    
    def test_convolution_mapping_pswf(self):
        
        self.actualSetUp()
        griddata = create_griddata_from_image(self.model)
        gcf, aaf = create_pswf_kernel(griddata, oversampling=8)
        self.vis = predict_skycomponent_visibility(self.vis, self.components)
        griddata, sumwt = grid_visibility_to_griddata(self.vis, griddata, aaf)
        im_griddata = convert_griddata_to_image(griddata)
        im_griddata.data = numpy.real(im_griddata.data)
        export_image_to_fits(im_griddata, '%s/test_gridding_griddata_pswf.fits' % self.dir)


    def test_convolution_mapping_wterm(self):
        self.actualSetUp()
        griddata = create_griddata_from_image(self.model)
        gcf, aaf = create_wterm_kernel(griddata, nw=20, wstep=10.0, oversampling=8)
        convolution_mapping(self.vis, griddata, aaf)


if __name__ == '__main__':
    unittest.main()
