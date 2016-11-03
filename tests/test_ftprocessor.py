"""Unit tests for Fourier transforms

realtimcornwell@gmail.com
"""
import logging
import unittest

import numpy

from arl.fourier_transforms.ftprocessor import *
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.skymodel.operations import create_skycomponent, find_skycomponent, fit_skycomponent
from arl.util.testing_support import create_named_configuration, create_test_image
from arl.visibility.operations import create_visibility
from arl.image.operations import export_image_to_fits

log = logging.getLogger("tests.test_ftprocessor")


class TestFTProcessor(unittest.TestCase):
    def setUp(self):
        
        self.params = {'wstep': 10.0, 'npixel': 512, 'cellsize': 0.1, 'spectral_mode': 'channel',
                       'channelwidth':5e7, 'reffrequency':1e8}
        
        self.field_of_view = self.params['npixel'] * self.params['cellsize']
        self.uvmax = 0.3 / self.params['cellsize']
        
        self.vlaa = create_named_configuration('VLAA')
        self.times = numpy.arange(-3.0, +3.0, 6.0 / 60.0) * numpy.pi / 12.0
        self.frequency = numpy.array([1e8, 1.5e8, 2e8])
 
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        self.average = numpy.average(self.flux[:, 0])
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=+35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.compabsdirection = SkyCoord(ra=15.05 * u.deg, dec=+35.10 * u.deg, frame='icrs', equinox=2000.0)
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)
        self.comp = create_skycomponent(flux=self.flux, frequency=self.frequency, direction=self.compreldirection)
        self.vis = create_visibility(self.vlaa, self.times, self.frequency, weight=1.0, phasecentre=self.phasecentre,
                                     params=self.params)
        self.vis.data['uvw'][:,2]=0.0
        self.componentvis = create_visibility(self.vlaa, self.times, self.frequency, weight=1.0,
                                              phasecentre=self.phasecentre, params=self.params)
        # Predict the visibility using direct evaluation
        self.componentvis = predict_skycomponent_visibility(self.componentvis, self.comp)

        self.model = create_image_from_visibility(self.vis)
        self.dirty = create_image_from_visibility(self.vis)
        self.psf   = create_image_from_visibility(self.vis)

    def test_predict_kernel(self):
        for ftpfunc in [predict_2d]:
            log.debug("ftpfunc %s" % ftpfunc)
            self.vis=ftpfunc(model=self.model, vis=self.vis, kernel=None, params=self.params)
            print (numpy.sum(abs(self.vis.data['vis']*numpy.conj(self.componentvis.data['vis']))))
            print (numpy.sum(abs(self.vis.data['weight']*numpy.conj(self.componentvis.data['weight']))))

    
    def test_invert_2d(self):
        # Set all w to zero
        self.vis.data['uvw'][:, 2] = 0.0
        self.componentvis = create_visibility(self.vlaa, self.times, self.frequency, weight=1.0,
                                              phasecentre=self.phasecentre,
                                              params=self.params)
        # Predict the visibility using direct evaluation
        self.componentvis = predict_skycomponent_visibility(self.componentvis, self.comp)
    
        self.dirty = invert_2d(vis=self.componentvis, im=self.model, dopsf=False, kernel=None, params=self.params)
        self.psf   = invert_2d(vis=self.componentvis, im=self.model, dopsf=True, kernel=None, params=self.params)
        psfmax = self.psf.data.max()
        print (psfmax, numpy.sum(self.vis.data['weight']))
        self.dirty.data /= psfmax
        self.psf.data   /= psfmax
        export_image_to_fits(self.dirty, 'test_invert_kernel_dirty.fits')
        export_image_to_fits(self.psf,   'test_invert_kernel_psf.fits')

        for comp in (self.comp, find_skycomponent(self.dirty, self.params), find_skycomponent(self.psf,
                                                                                              self.params)):
            print(comp.flux, comp.direction, comp.frequency)

    def test_invert_kernel(self):
        for ftpfunc in [invert_2d]:
            log.debug("ftpfunc %s" % ftpfunc)
            self.dirty = ftpfunc(vis=self.componentvis, im=self.model, dopsf=False, kernel=None, params=self.params)
            self.psf   = ftpfunc(vis=self.componentvis, im=self.model, dopsf=True, kernel=None, params=self.params)
            psfmax = self.psf.data.max()
            print (psfmax, numpy.sum(self.vis.data['weight']))
            self.dirty.data /= psfmax
            self.psf.data   /= psfmax
            export_image_to_fits(self.dirty, 'test_invert_kernel_dirty.fits')
            export_image_to_fits(self.psf,   'test_invert_kernel_psf.fits')

            for comp in (self.comp, find_skycomponent(self.dirty, self.params), find_skycomponent(self.psf,
                                                                                                  self.params)):
                print(comp.flux, comp.direction, comp.frequency)

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
