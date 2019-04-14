import sys
import unittest

from data_models.parameters import arl_path

import numpy

from wrappers.serial.visibility.base import create_blockvisibility_from_ms
from wrappers.serial.image.operations import export_image_to_fits, qa_image
from wrappers.serial.image.deconvolution import deconvolve_cube, restore_cube
from wrappers.serial.imaging.base import create_image_from_visibility
from wrappers.serial.visibility.coalesce import convert_blockvisibility_to_visibility
from wrappers.serial.visibility.operations import convert_visibility_to_stokes

from workflows.serial.imaging.imaging_serial import invert_list_serial_workflow

from data_models.polarisation import PolarisationFrame

import logging
log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImagingSim2(unittest.TestCase):
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
    
    def tearDown(self):
        pass
    
    def test_sim2(self):
        
        try:
            bvt = create_blockvisibility_from_ms(arl_path('data/vis/sim-2.ms'), channum=[35, 36, 37, 38, 39])[0]
            bvt.configuration.diameter[...] = 35.0
            vt = convert_blockvisibility_to_visibility(bvt)
            vt = convert_visibility_to_stokes(vt)
            
            cellsize = 20.0 * numpy.pi / (180.0 * 3600.0)
            npixel = 512
            
            model = create_image_from_visibility(vt, cellsize=cellsize, npixel=npixel,
                                                 polarisation_frame=PolarisationFrame('stokesIQUV'))
            dirty, sumwt = invert_list_serial_workflow([vt], [model], context='2d')[0]
            psf, sumwt = invert_list_serial_workflow([vt], [model], context='2d', dopsf=True)[0]
            export_image_to_fits(dirty, '%s/imaging_sim2_dirty.fits' % (self.dir))
            export_image_to_fits(psf, '%s/imaging_sim2_psf.fits' % (self.dir))
            
            # Deconvolve using clean
            comp, residual = deconvolve_cube(dirty, psf, niter=10000, threshold=0.001, fractional_threshold=0.001,
                                             window_shape='quarter', gain=0.7, scales=[0, 3, 10, 30])
            
            restored = restore_cube(comp, psf, residual)
            export_image_to_fits(restored, '%s/imaging_sim2_restored.fits' % (self.dir))
            export_image_to_fits(residual, '%s/imaging_sim2_residual.fits' % (self.dir))
            
            qa = qa_image(restored)
            
            assert numpy.abs(qa.data['max'] - 1.006140596404203) < 1e-7, qa
            assert numpy.abs(qa.data['maxabs'] - 1.006140596404203) < 1e-7, qa
            assert numpy.abs(qa.data['min'] + 0.23890808520954754) < 1e-7, qa
            assert numpy.abs(qa.data['rms'] - 0.007366519782047875) < 1e-7, qa
            assert numpy.abs(qa.data['medianabs'] - 0.0005590537883509844) < 1e-7, qa
        except ModuleNotFoundError:
            pass


if __name__ == '__main__':
    unittest.main()
