""" Unit tests for image operations


"""
import logging
import unittest

import numpy

from astropy.coordinates import SkyCoord
import astropy.units as u

from processing_components.griddata.kernels import create_pswf_kernel, create_wterm_kernel, create_awterm_kernel
from processing_components.griddata.operations import create_griddata_from_image, convert_griddata_to_image
from processing_components.imaging.primary_beams import create_pb_generic
from libs.image.operations import create_image
from processing_components.image.operations import export_image_to_fits

log = logging.getLogger(__name__)


class TestGridDataKernels(unittest.TestCase):
    
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.image = create_image(npixel=512, cellsize=0.001, phasecentre=self.phasecentre)
        self.griddata = create_griddata_from_image(self.image)
    
    def test_fill_aaf_to_griddata(self):
        gcf, aaf = create_pswf_kernel(self.griddata)
        assert numpy.max(numpy.abs(aaf.data)) > 0.0
        export_image_to_fits(gcf, "%s/test_griddata_pswf_cf.fits" % self.dir)
        convfunc = convert_griddata_to_image(aaf)
        convfunc.data = numpy.real(convfunc.data)
        export_image_to_fits(convfunc, "%s/test_griddata_pswf_convfunc.fits" % self.dir)
    
    def test_fill_wterm_to_griddata(self):
        self.griddata = create_griddata_from_image(self.image)
        nchan, npol, nz, ny, nx = self.griddata.shape
        nw = 20
        wstep = 100.0
        self.griddata.data = numpy.zeros([nchan, npol, nw, ny, nx], dtype='complex')
        self.griddata.grid_wcs.wcs.crpix[2] = nw / 2.0
        self.griddata.grid_wcs.wcs.cdelt[2] = wstep
        gcf, aaf = create_wterm_kernel(self.griddata, use_aaf=True)
        assert numpy.max(numpy.abs(aaf.data)) > 0.0
        export_image_to_fits(gcf, "%s/test_griddata_wterm_cf.fits" % self.dir)
        convfunc = convert_griddata_to_image(aaf)
        convfunc.data = numpy.real(convfunc.data)
        export_image_to_fits(convfunc, "%s/test_griddata_wterm_convfunc.fits" % self.dir)
    
    def test_fill_awterm_to_griddata(self):
        self.griddata = create_griddata_from_image(self.image)
        nchan, npol, nz, ny, nx = self.griddata.shape
        nw = 20
        wstep = 100.0
        self.griddata.data = numpy.zeros([nchan, npol, nw, ny, nx], dtype='complex')
        self.griddata.grid_wcs.wcs.crpix[2] = nw / 2.0
        self.griddata.grid_wcs.wcs.cdelt[2] = wstep
        pb = create_pb_generic(self.image, diameter=35.0, blockage=0.0)
        gcf, aaf = create_awterm_kernel(self.griddata, pb=pb, use_aaf=True)
        assert numpy.max(numpy.abs(aaf.data)) > 0.0
        export_image_to_fits(pb, "%s/test_griddata_awterm_pb.fits" % self.dir)
        export_image_to_fits(gcf, "%s/test_griddata_awterm_cf.fits" % self.dir)
        convfunc = convert_griddata_to_image(aaf)
        convfunc.data = numpy.real(convfunc.data)
        export_image_to_fits(convfunc, "%s/test_griddata_awterm_convfunc.fits" % self.dir)


if __name__ == '__main__':
    unittest.main()
