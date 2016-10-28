"""Unit tests for Fourier transforms

realtimcornwell@gmail.com
"""
import logging
import unittest

import numpy

from arl.fourier_transforms.ftprocessor import *
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.image.image_operations import create_empty_image_like
from arl.skymodel.skymodel_operations import create_skycomponent
from arl.skymodel.skymodel_operations import create_skymodel_from_component
from arl.util.testing_support import create_named_configuration, create_test_image
from arl.visibility.visibility_operations import create_visibility

log = logging.getLogger("tests.test_ftprocessor")


class TestFTProcessor(unittest.TestCase):
    def setUp(self):
        
        self.params = {'wstep': 10.0, 'npixel': 512, 'cellsize': 0.004, 'spectral_mode': 'channel'}
        
        self.field_of_view = self.params['npixel'] * self.params['cellsize']
        self.uvmax = 0.3 / self.params['cellsize']
        
        vlaa = create_named_configuration('VLAA')
        times = numpy.arange(-3.0, +3.0, 6.0 / 60.0) * numpy.pi / 12.0
        frequency = numpy.arange(1.0e8, 1.50e8, 2.0e7)
        
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        self.average = numpy.average(self.flux[:, 0])
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=+35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=+36.5 * u.deg, frame='icrs', equinox=2000.0)
        # TODO: convert entire mechanism to absolute coordinates
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)
        self.comp = create_skycomponent(flux=self.flux, frequency=frequency, direction=self.compreldirection)
        self.sm = create_skymodel_from_component(self.comp)
        self.vis = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=self.phasecentre,
                                     params=self.params)
        self.model = create_test_image(npol=4, nchan=3)
        # Scale the image in x, y appropriately
        self.model.wcs.wcs.cdelt[0:1] *= 5.0
        self.dirty = create_empty_image_like(self.model)
        self.psf = create_empty_image_like(self.model)
        
        self.vis = predict_2d(self.vis, self.model)
    
    def test_predict_kernel(self):
        for ftpfunc in [predict_2d]:
            log.debug("ftpfunc %s" % ftpfunc)
            ftpfunc(model=self.model, vis=self.vis, kernel=None, params=self.params)
    
    def test_invert_kernel(self):
        sumofweights = 0.0
        for ftpfunc in [invert_2d]:
            log.debug("ftpfunc %s" % ftpfunc)
            result = ftpfunc(vis=self.vis, im=self.model, dopsf=False, kernel=None,
                             params=self.params)
            self.dirty.data = result
            
            result = ftpfunc(vis=self.vis, im=self.model, dopsf=True, kernel=None,
                             params=self.params)
            self.psf.data = result
    
    @unittest.skip('Predict image partition not yet working')  # Need to sort out coordinate and cellsizes
    def test_predict_partition(self):
        for ftpfunc in [predict_image_partition]:
            # [predict_wslice_partition, predict_image_partition, predict_fourier_partition]:
            log.debug("ftpfunc %s" % ftpfunc)
            ftpfunc(model=self.model, vis=self.vis, predict_function=predict_2d, params=self.params)
    
    def test_invert_partition(self):
        for ftpfunc in [invert_image_partition]:
            # [invert_wslice_partition, invert_image_partition, invert_fourier_partition]:
            log.debug("ftpfunc %s" % ftpfunc)
            result = ftpfunc(vis=self.vis, im=self.model, dopsf=False, kernel=None,
                             invert_function=invert_2d, params=self.params)
            self.dirty.data = result
            
            result = ftpfunc(vis=self.vis, im=self.model, dopsf=True,
                             invert_function=invert_2d, params=self.params)
            self.psf.data = result


if __name__ == '__main__':
    import sys
    import logging
    
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    unittest.main()
