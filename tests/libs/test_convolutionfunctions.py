""" Unit tests for image operations


"""
import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from libs.image.operations import create_image
from processing_components.convolution_function.kernels import create_pswf_convolutionfunction, \
    create_awterm_convolutionfunction
from processing_components.convolution_function.operations import convert_convolutionfunction_to_image, \
    create_convolutionfunction_from_image, apply_bounding_box_convolutionfunction, \
    calculate_bounding_box_convolutionfunction
from processing_components.image.operations import export_image_to_fits
from processing_components.imaging.primary_beams import create_pb_generic

log = logging.getLogger(__name__)


class TestGridDataKernels(unittest.TestCase):
    
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.image = create_image(npixel=512, cellsize=0.0005, phasecentre=self.phasecentre)
    
    def test_create_convolutionfunction(self):
        cf = create_convolutionfunction_from_image(self.image, nz=1, zstep=1e-7)
        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_cf.fits" % self.dir)
    
    def test_fill_pswf_to_convolutionfunction(self):
        gcf, cf = create_pswf_convolutionfunction(self.image, oversampling=16, support=6)
        assert numpy.max(numpy.abs(cf.data)) > 0.0
        export_image_to_fits(gcf, "%s/test_convolutionfunction_pswf_gcf.fits" % self.dir)
        
        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_pwsf_cf.fits" % self.dir)

    def test_fill_wterm_to_convolutionfunction(self):
        gcf, cf = create_awterm_convolutionfunction(self.image, pb=None, nw=5, wstep=100.0, use_aaf=True,
                                                    oversampling=4, support=256)
        assert numpy.max(numpy.abs(cf.data)) > 0.0
        export_image_to_fits(gcf, "%s/test_convolutionfunction_wterm_gcf.fits" % self.dir)
    
        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_wterm_cf.fits" % self.dir)

        cf_clipped = apply_bounding_box_convolutionfunction(cf)
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf_clipped.data)), cf_clipped.shape)
        assert peak_location == (0, 0, 2, 0, 0, 128, 128), peak_location
        cf_image = convert_convolutionfunction_to_image(cf_clipped)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_wterm_clipped_cf.fits" % self.dir)

    def test_fill_wterm_to_convolutionfunction_nopswf(self):
        gcf, cf = create_awterm_convolutionfunction(self.image, pb=None, nw=5, wstep=100.0, use_aaf=False,
                                                    oversampling=4, support=256)
        assert numpy.max(numpy.abs(cf.data)) > 0.0
        export_image_to_fits(gcf, "%s/test_convolutionfunction_wterm_nopswf_cf.fits" % self.dir)
    
        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_wterm_nopswf_cf.fits" % self.dir)

    def test_fill_awterm_to_convolutionfunction(self):
        pb = create_pb_generic(self.image, diameter=35.0, blockage=0.0)
        export_image_to_fits(pb, "%s/test_convolutionfunction_awterm_pb.fits" % self.dir)
        gcf, cf = create_awterm_convolutionfunction(self.image, pb=pb, nw=5, wstep=100.0, use_aaf=True,
                                                    oversampling=4, support=256)
        
        assert numpy.max(numpy.abs(cf.data)) > 0.0
        export_image_to_fits(gcf, "%s/test_convolutionfunction_awterm_gcf.fits" % self.dir)
        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_awterm_cf.fits" % self.dir)
        
        bboxes = calculate_bounding_box_convolutionfunction(cf)
        assert len(bboxes) == 5
        assert len(bboxes[0]) == 3
        assert bboxes[-1][0] == 4
        
        cf_clipped = apply_bounding_box_convolutionfunction(cf)
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf_clipped.data)), cf_clipped.shape)
        assert peak_location == (0, 0, 2, 0, 0, 117, 117), peak_location
        cf_image = convert_convolutionfunction_to_image(cf_clipped)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_awterm_clipped_cf.fits" % self.dir)

    
    def test_fill_aterm_to_convolutionfunction(self):
        pb = create_pb_generic(self.image, diameter=35.0, blockage=0.0)
        
        export_image_to_fits(pb, "%s/test_convolutionfunction_aterm_pb.fits" % self.dir)
        gcf, cf = create_awterm_convolutionfunction(self.image, pb=pb, nw=1, wstep=1e-7, use_aaf=True,
                                                    oversampling=4, support=16)
        
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data)), cf.shape)
        assert peak_location == (0, 0, 0, 0, 0, 8, 8), peak_location
        assert numpy.max(numpy.abs(cf.data)) > 0.0
        export_image_to_fits(gcf, "%s/test_convolutionfunction_aterm_gcf.fits" % self.dir)
        
        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_aterm_cf.fits" % self.dir)

        cf_clipped = apply_bounding_box_convolutionfunction(cf)
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf_clipped.data)), cf_clipped.shape)
        assert peak_location == (0, 0, 0, 0, 0, 4, 4), peak_location
        cf_image = convert_convolutionfunction_to_image(cf_clipped)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_aterm_clipped_cf.fits" % self.dir)

if __name__ == '__main__':
    unittest.main()
