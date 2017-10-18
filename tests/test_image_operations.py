"""Unit tests for image operations


"""
import logging
import os
import unittest

import numpy

from arl.data.polarisation import PolarisationFrame
from arl.image.operations import copy_image, create_empty_image_like, create_image_from_array, add_image, \
    export_image_to_fits, qa_image, reproject_image, smooth_image, checkwcs, convert_polimage_to_stokes, \
    convert_stokes_to_polimage, polarisation_frame_from_wcs, fft_image, show_image, \
    calculate_image_frequency_moments, calculate_image_from_frequency_moments, pad_image, convert_image_to_kernel, \
    create_w_term_like
from arl.util.testing_support import create_test_image, create_low_test_image_from_gleam

log = logging.getLogger(__name__)

class TestImage(unittest.TestCase):

    def setUp(self):
    
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
    
        self.m31image = create_test_image(cellsize=0.0001)
        self.cellsize = 180.0 * 0.0001 / numpy.pi

    def test_create_image_from_array(self):
        m31model_by_array = create_image_from_array(self.m31image.data, wcs=None)
        
        m31model_by_array = create_image_from_array(self.m31image.data, self.m31image.wcs)
        m31modelsum = add_image(self.m31image, m31model_by_array)
        m31modelsum = add_image(self.m31image, m31model_by_array, docheckwcs=True)
        assert m31model_by_array.shape == self.m31image.shape
        log.debug(export_image_to_fits(self.m31image, fitsfile='%s/test_model.fits' % (self.dir)))
        log.debug(qa_image(m31model_by_array, context='test_create_from_image'))

    def test_create_empty_image_like(self):
        emptyimage = create_empty_image_like(self.m31image)
        assert emptyimage.shape == self.m31image.shape
        assert numpy.max(numpy.abs(emptyimage.data)) == 0.0

    def test_checkwcs(self):
    
        newwcs = self.m31image.wcs.deepcopy()
        newwcs = self.m31image.wcs
        checkwcs(self.m31image.wcs, newwcs)
        cellsize = 1.5 * self.cellsize
        newwcs.wcs.cdelt[0] = -cellsize
        newwcs.wcs.cdelt[1] = +cellsize
        # with self.assertRaises(AssertionError):
        #     checkwcs(self.m31image.wcs, newwcs)
    
    def test_reproject(self):
        # Reproject an image
    
        cellsize = 1.5 * self.cellsize
        newwcs = self.m31image.wcs.deepcopy()
        newwcs.wcs.cdelt[0] = -cellsize
        newwcs.wcs.cdelt[1] = +cellsize
    
        newshape = numpy.array(self.m31image.data.shape)
        newshape[2] /= 1.5
        newshape[3] /= 1.5
        newimage, footprint = reproject_image(self.m31image, newwcs, shape=newshape)
        checkwcs(newimage.wcs, newwcs)

    def test_stokes_conversion(self):
        assert self.m31image.polarisation_frame == PolarisationFrame("stokesI")
        stokes = create_test_image(cellsize=0.0001, polarisation_frame=PolarisationFrame("stokesIQUV"))
        assert stokes.polarisation_frame == PolarisationFrame("stokesIQUV")

        for pol_name in ['circular', 'linear']:
            polarisation_frame = PolarisationFrame(pol_name)
            polimage = convert_stokes_to_polimage(stokes, polarisation_frame=polarisation_frame)
            assert polimage.polarisation_frame == polarisation_frame
            pf = polarisation_frame_from_wcs(polimage.wcs, polimage.shape)
            rstokes = convert_polimage_to_stokes(polimage)
            assert polimage.data.dtype == 'complex'
            assert rstokes.data.dtype == 'complex'
            numpy.testing.assert_array_almost_equal(stokes.data, rstokes.data.real, 12)
            
    def test_polarisation_frame_from_wcs(self):
        assert self.m31image.polarisation_frame == PolarisationFrame("stokesI")
        stokes = create_test_image(cellsize=0.0001, polarisation_frame=PolarisationFrame("stokesIQUV"))
        wcs = stokes.wcs.deepcopy()
        shape = stokes.shape
        assert polarisation_frame_from_wcs(wcs, shape) == PolarisationFrame("stokesIQUV")
        
        wcs = stokes.wcs.deepcopy().sub(['stokes'])
        wcs.wcs.crpix[0] = 1.0
        wcs.wcs.crval[0] = -1.0
        wcs.wcs.cdelt[0] = -1.0
        assert polarisation_frame_from_wcs(wcs, shape) == PolarisationFrame('circular')

        wcs.wcs.crpix[0] = 1.0
        wcs.wcs.crval[0] = -5.0
        wcs.wcs.cdelt[0] = -1.0
        assert polarisation_frame_from_wcs(wcs, shape) == PolarisationFrame('linear')

        wcs.wcs.crpix[0] = 1.0
        wcs.wcs.crval[0] = -1.0
        wcs.wcs.cdelt[0] = -1.0
        assert polarisation_frame_from_wcs(wcs, shape) == PolarisationFrame('circular')

        with self.assertRaises(ValueError):
            wcs.wcs.crpix[0] = 1.0
            wcs.wcs.crval[0] = -100.0
            wcs.wcs.cdelt[0] = -1.0
            pf = polarisation_frame_from_wcs(wcs, shape)


    def test_smooth_image(self):
        smooth = smooth_image(self.m31image)
        assert numpy.max(smooth.data) > numpy.max(self.m31image.data)
        
    def test_calculate_image_frequency_moments(self):
        frequency = numpy.linspace(0.9e8, 1.1e8, 9)
        cube=create_low_test_image_from_gleam(npixel=512, cellsize=0.0001, frequency=frequency)
        log.debug(export_image_to_fits(cube, fitsfile='%s/test_moments_cube.fits' % (self.dir)))
        original_cube = copy_image(cube)
        moment_cube=calculate_image_frequency_moments(cube, nmoments=3)
        log.debug(export_image_to_fits(moment_cube, fitsfile='%s/test_moments_moment_cube.fits' % (self.dir)))
        reconstructed_cube=calculate_image_from_frequency_moments(cube, moment_cube)
        log.debug(export_image_to_fits(reconstructed_cube, fitsfile='%s/test_moments_reconstructed_cube.fits' % (
            self.dir)))
        error = numpy.std(reconstructed_cube.data-original_cube.data)
        assert error < 0.2

    def test_create_w_term_image(self):
        m31image = create_test_image(cellsize=0.001)
        im = create_w_term_like(m31image, w=20000.0, remove_shift=True)
        im.data = im.data.real
        for x in [64, 64 + 128]:
            for y in [64, 64 + 128]:
                self.assertAlmostEqual(im.data[0, 0, y, x], 0.84946344276442431, 7)
        export_image_to_fits(im, '%s/test_wterm.fits' % self.dir)
        assert im.data.shape == (1, 1, 256, 256)
        self.assertAlmostEqual(numpy.max(im.data.real), 1.0, 7)

    def test_fftim(self):
        self.m31image = create_test_image(cellsize=0.001, frequency=[1e8], canonical=True)
        m31_fft = fft_image(self.m31image)
        m31_fft_ifft = fft_image(m31_fft, self.m31image)
        numpy.testing.assert_array_almost_equal(self.m31image.data, m31_fft_ifft.data.real, 12)
        m31_fft.data = numpy.abs(m31_fft.data)
        export_image_to_fits(m31_fft, fitsfile='%s/test_m31_fft.fits' % (self.dir))

    def test_fftim_factors(self):
        for i in [3, 5, 7]:
            npixel=256 * i
            m31image = create_test_image(cellsize=0.001, frequency=[1e8], canonical=True)
            padded = pad_image(m31image, [1, 1, npixel, npixel])
            assert padded.shape == (1, 1, npixel, npixel)
            padded_fft = fft_image(padded)
            padded_fft_ifft = fft_image(padded_fft, m31image)
            numpy.testing.assert_array_almost_equal(padded.data, padded_fft_ifft.data.real, 12)
            padded_fft.data = numpy.abs(padded_fft.data)
            export_image_to_fits(padded_fft, fitsfile='%s/test_m31_fft_%d.fits' % (self.dir, npixel))

    def test_pad_image(self):
        m31image = create_test_image(cellsize=0.001, frequency=[1e8], canonical=True)
        padded = pad_image(m31image, [1, 1, 1024, 1024])
        assert padded.shape == (1, 1, 1024, 1024)
        
        padded = pad_image(m31image, [3, 4, 2048, 2048])
        assert padded.shape == (3, 4, 2048, 2048)
        
        with self.assertRaises(ValueError):
            padded = pad_image(m31image, [1, 1, 100, 100])
            
        with self.assertRaises(IndexError):
            padded = pad_image(m31image, [1, 1])
            
    def test_convert_image_to_kernel(self):
        m31image = create_test_image(cellsize=0.001, frequency=[1e8], canonical=True)
        screen = create_w_term_like(m31image, w=20000.0, remove_shift=True)
        screen_fft = fft_image(screen)
        converted = convert_image_to_kernel(screen_fft, 8, 8)
        assert converted.shape == (1, 1, 8, 8, 8, 8)
        with self.assertRaises(AssertionError):
            converted = convert_image_to_kernel(m31image, 15, 1)
        with self.assertRaises(AssertionError):
            converted = convert_image_to_kernel(m31image, 15, 1000)

if __name__ == '__main__':
    unittest.main()
