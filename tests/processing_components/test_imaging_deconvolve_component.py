""" Unit tests for pipelines expressed via dask.delayed


"""

import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame

from processing_components.imaging.imaging_components import invert_component, deconvolve_component, \
    residual_component, restore_component
from libs.execution_support.arlexecute import arlexecute
from processing_components.image.operations import export_image_to_fits, smooth_image
from processing_components.imaging.base import predict_skycomponent_visibility
from processing_components.skycomponent.operations import insert_skycomponent
from simulation.testing_support import create_named_configuration, ingest_unittest_visibility, \
    create_unittest_model, create_unittest_components, insert_unittest_errors

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImagingDeconvolveGraph(unittest.TestCase):
    
    def setUp(self):
        
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        arlexecute.set_client(use_dask=False)
    
    def tearDown(self):
        arlexecute.close()
    
    def actualSetUp(self, add_errors=False, freqwin=7, block=False, dospectral=True, dopol=False,
                    zerow=True):
        
        self.npixel = 256
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = freqwin
        self.vis_list = list()
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        
        if freqwin > 1:
            self.channelwidth = numpy.array(freqwin * [self.frequency[1] - self.frequency[0]])
        else:
            self.channelwidth = numpy.array([1e6])
        
        if dopol:
            self.vis_pol = PolarisationFrame('linear')
            self.image_pol = PolarisationFrame('stokesIQUV')
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        else:
            self.vis_pol = PolarisationFrame('stokesI')
            self.image_pol = PolarisationFrame('stokesI')
            f = numpy.array([100.0])
        
        if dospectral:
            flux = numpy.array([f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency])
        else:
            flux = numpy.array([f])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis_list = [arlexecute.execute(ingest_unittest_visibility)(self.low,
                                                                        [self.frequency[freqwin]],
                                                                        [self.channelwidth[freqwin]],
                                                                        self.times,
                                                                        self.vis_pol,
                                                                        self.phasecentre, block=block,
                                                                        zerow=zerow)
                         for freqwin, _ in enumerate(self.frequency)]
        
        self.model_imagelist = [arlexecute.execute(create_unittest_model, nout=freqwin)(self.vis_list[freqwin],
                                                                                        self.image_pol,
                                                                                        npixel=self.npixel)
                                for freqwin, _ in enumerate(self.frequency)]
        
        self.componentlist = [arlexecute.execute(create_unittest_components)(self.model_imagelist[freqwin],
                                                                             flux[freqwin, :][numpy.newaxis, :])
                              for freqwin, _ in enumerate(self.frequency)]
        
        self.model_imagelist = [arlexecute.execute(insert_skycomponent, nout=1)(self.model_imagelist[freqwin],
                                                                                self.componentlist[freqwin])
                                for freqwin, _ in enumerate(self.frequency)]
        
        self.vis_list = [arlexecute.execute(predict_skycomponent_visibility)(self.vis_list[freqwin],
                                                                             self.componentlist[freqwin])
                         for freqwin, _ in enumerate(self.frequency)]
        
        # Calculate the model convolved with a Gaussian.
        
        model = arlexecute.compute(self.model_imagelist[0], sync=True)
        
        self.cmodel = smooth_image(model)
        export_image_to_fits(model, '%s/test_imaging_delayed_deconvolved_model.fits' % self.dir)
        export_image_to_fits(self.cmodel, '%s/test_imaging_deconvolved_delayed_cmodel.fits' % self.dir)
        
        if add_errors and block:
            self.vis_list = [arlexecute.execute(insert_unittest_errors)(self.vis_list[i])
                             for i, _ in enumerate(self.frequency)]
    
    def test_time_setup(self):
        self.actualSetUp()
    
    def test_deconvolve_spectral(self):
        self.actualSetUp(add_errors=True)
        dirty_imagelist = invert_component(self.vis_list, self.model_imagelist,
                                           context='2d',
                                           dopsf=False, normalize=True)
        psf_imagelist = invert_component(self.vis_list, self.model_imagelist,
                                         context='2d',
                                         dopsf=True, normalize=True)
        deconvolved, _ = deconvolve_component(dirty_imagelist, psf_imagelist, self.model_imagelist, niter=1000,
                                              fractional_threshold=0.1, scales=[0, 3, 10],
                                              threshold=0.1, gain=0.7)
        deconvolved = arlexecute.compute(deconvolved, sync=True)
        
        export_image_to_fits(deconvolved[0], '%s/test_imaging_%s_deconvolve_spectral.fits' %
                             (self.dir, arlexecute.type()))
    
    def test_deconvolve_and_restore_cube_mmclean(self):
        self.actualSetUp(add_errors=True)
        dirty_imagelist = invert_component(self.vis_list, self.model_imagelist, context='2d',
                                           dopsf=False, normalize=True)
        psf_imagelist = invert_component(self.vis_list, self.model_imagelist, context='2d',
                                         dopsf=True, normalize=True)
        dec_imagelist, _ = deconvolve_component(dirty_imagelist, psf_imagelist, self.model_imagelist, niter=1000,
                                                fractional_threshold=0.01, scales=[0, 3, 10],
                                                algorithm='mmclean', nmoments=3, nchan=self.freqwin,
                                                threshold=0.1, gain=0.7)
        residual_imagelist = residual_component(self.vis_list, model_imagelist=dec_imagelist,
                                                context='wstack', vis_slices=51)
        restored = restore_component(model_imagelist=dec_imagelist, psf_imagelist=psf_imagelist,
                                     residual_imagelist=residual_imagelist,
                                     empty=self.model_imagelist)[0]
        
        restored = arlexecute.compute(restored, sync=True)
        
        export_image_to_fits(restored, '%s/test_imaging_%s_mmclean_restored.fits' % (self.dir, arlexecute.type()))
    
    def test_deconvolve_and_restore_cube_mmclean_facets(self):
        self.actualSetUp(add_errors=True)
        dirty_imagelist = invert_component(self.vis_list, self.model_imagelist,
                                           context='2d', dopsf=False, normalize=True)
        psf_imagelist = invert_component(self.vis_list, self.model_imagelist,
                                         context='2d', dopsf=True, normalize=True)
        dec_imagelist, _ = deconvolve_component(dirty_imagelist, psf_imagelist, self.model_imagelist, niter=1000,
                                                fractional_threshold=0.1, scales=[0, 3, 10],
                                                algorithm='mmclean', nmoments=3, nchan=self.freqwin,
                                                threshold=0.01, gain=0.7, deconvolve_facets=8,
                                                deconvolve_overlap=8, deconvolve_taper='tukey')
        residual_imagelist = residual_component(self.vis_list, model_imagelist=dec_imagelist,
                                                context='2d')
        restored = restore_component(model_imagelist=dec_imagelist, psf_imagelist=psf_imagelist,
                                     residual_imagelist=residual_imagelist,
                                     empty=self.model_imagelist)[0]
        
        restored = arlexecute.compute(restored, sync=True)
        
        export_image_to_fits(restored, '%s/test_imaging_%s_overlap_mmclean_restored.fits'
                             % (self.dir, arlexecute.type()))


if __name__ == '__main__':
    unittest.main()
