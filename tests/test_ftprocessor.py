"""Unit tests for Fourier transforms

realtimcornwell@gmail.com
"""
import unittest

from astropy.convolution import Gaussian2DKernel, convolve

from arl.fourier_transforms.ftprocessor import *
from arl.image.operations import export_image_to_fits, create_empty_image_like
from arl.skycomponent.operations import create_skycomponent, find_skycomponents, find_nearest_component, \
    insert_skycomponent
from arl.util.testing_support import create_named_configuration
import logging

from arl.visibility.operations import create_visibility, create_blockvisibility, sum_visibility

log = logging.getLogger(__name__)


class TestFTProcessor(unittest.TestCase):
    def _checkdirty(self, vis, name='test_invert_2d_dirty', fluxthreshold=1.0):
        # Make the dirty image
        self.params['imaginary'] = False
        dirty = create_empty_image_like(self.model)
        dirty, sumwt = invert_2d(vis=vis, im=dirty, dopsf=False, **self.params)
        dirty = normalize_sumwt(dirty, sumwt)
        export_image_to_fits(dirty, '%s/%s_dirty.fits' % (self.dir, name))
        maxabs = numpy.max(numpy.abs(dirty.data))
        assert maxabs < fluxthreshold, "%s, abs max %f exceeds flux threshold" % (name, maxabs)
    
    def _checkcomponents(self, dirty, fluxthreshold=20.0, positionthreshold=1.0):
        comps = find_skycomponents(dirty, fwhm=1.0, threshold=fluxthreshold, npixels=5)
        assert len(comps) == len(self.components), "Different number of components found: original %d, recovered %d" % \
                                                   (len(self.components), len(comps))
        cellsize = abs(dirty.wcs.wcs.cdelt[0])
        # Check for agreement between image and DFT
        for comp in comps:
            sflux = sum_visibility(self.componentvis, comp.direction)[0]
            assert abs(comp.flux[0, 0] - sflux[0, 0]) < fluxthreshold, \
                "Fitted and DFT flux differ %s %s" % (comp.flux[0, 0], sflux[0, 0])
            # Check for agreement in direction
            ocomp = find_nearest_component(comp.direction, self.components)
            radiff = abs(comp.direction.ra.deg - ocomp.direction.ra.deg) / cellsize
            assert radiff  < positionthreshold, "Component differs in dec %.3f pixels" % radiff
            decdiff = abs(comp.direction.dec.deg - ocomp.direction.dec.deg) / cellsize
            assert decdiff  < positionthreshold, "Component differs in dec %.3f pixels" % decdiff

    def setUp(self):
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        self.params = {'npixel': 256,
                       'nchan': 1,
                       'reffrequency': 1e8,
                       'facets': 8,
                       'padding': 2,
                       'oversampling': 4,
                       'timeslice': 'auto',
                       'wstep': 10.0,
                       'wslice': 10.0}

    def actualSetUp(self, time=None, frequency=None):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / (12.0)) * numpy.linspace(-3.0, 3.0, 7)
        
        if time is not None:
            self.times = time
        log.info("Times are %s" % (self.times))
        
        if frequency is None:
            self.frequency = numpy.array([1e8])
            self.channel_bandwidth = numpy.array([1e7])
        else:
            self.frequency = frequency
            if len(self.frequency) < 1:
                self.channel_bandwidth = numpy.full_like(self.frequency, self.frequency[1] - self.frequency[0])
            else:
                self.channel_bandwidth = numpy.array([1e6])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox=2000.0)
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency,
                                              channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                              weight=1.0, polarisation_frame=PolarisationFrame('stokesI'))
        self.uvw = self.componentvis.data['uvw']
        self.componentvis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_image_from_visibility(self.componentvis, npixel=256, cellsize=0.001,
                                                  nchan=1, polarisation_frame=PolarisationFrame('stokesI'))
        
        # Fill the visibility with exactly computed point sources. These are chosen to lie
        # on grid points.
        spacing_pixels = 32
        log.info('Spacing in pixels = %s' % spacing_pixels)
        
        centers = [-2.5, -0.5, 0.5, 2.5]
        
        # Make the list of components
        rpix = self.model.wcs.wcs.crpix - 1
        self.components = []
        for iy in centers:
            for ix in centers:
                if ix >= iy:
                    # The phase center in 0-relative coordinates is n // 2 so we centre the grid of
                    # components on ny // 2, nx // 2. The wcs must be defined consistently.
                    p = int(round(rpix[0] + ix * spacing_pixels * numpy.sign(self.model.wcs.wcs.cdelt[0]))), \
                        int(round(rpix[1] + iy * spacing_pixels * numpy.sign(self.model.wcs.wcs.cdelt[1])))
                    sc = pixel_to_skycoord(p[0], p[1], self.model.wcs, origin=0)
                    log.info("Component at (%f, %f) [0-rel] %s" % (p[0], p[1], str(sc)))
                    
                    f = (100.0 + 1.0 * ix + iy * 10.0)
                    # Channel images
                    flux = numpy.array([[f]])
                    comp = create_skycomponent(flux=flux, frequency=[numpy.average(self.frequency)], direction=sc,
                                               polarisation_frame=PolarisationFrame('stokesI'))
                    self.components.append(comp)
        
        # Predict the visibility from the components exactly. We always do this for each spectral channel
        self.componentvis.data['vis'] *= 0.0
        predict_skycomponent_visibility(self.componentvis, self.components)
        insert_skycomponent(self.model, self.components)

        # Calculate the model convolved with a Gaussian.
        norm = 2.0 * numpy.pi
        self.cmodel = copy_image(self.model)
        self.cmodel.data[0, 0, :, :] = norm * convolve(self.model.data[0, 0, :, :], Gaussian2DKernel(1.0),
                                                      normalize_kernel=False)
        export_image_to_fits(self.model, '%s/test_model.fits' % self.dir)
        export_image_to_fits(self.cmodel, '%s/test_cmodel.fits' % self.dir)

    def test_findcomponents(self):
        # Check that the components are where we expected them to be after insertion
        self.actualSetUp()
        self._checkcomponents(self.cmodel)


    def test_predict_2d(self):
        """Test if the 2D prediction works

        Set w=0 so that the two-dimensional transform should agree exactly with the component transform.
        Good check on the grid correction in the image->vis direction"""
        # Set all w to zero
        self.actualSetUp()
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency, channel_bandwidth = \
            self.channel_bandwidth, phasecentre=self.phasecentre, weight=1.0)
        self.componentvis.data['uvw'][:, 2] = 0.0
        # Predict the visibility using direct evaluation
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)
        
        self.modelvis = create_visibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                          weight=1.0, polarisation_frame=PolarisationFrame('stokesI'))
        self.modelvis.data['uvw'][:, 2] = 0.0
        predict_2d(self.modelvis, self.model, **self.params)
        self.residualvis = create_visibility(self.lowcore, self.times, self.frequency,
                                             channel_bandwidth=self.channel_bandwidth,
                                             phasecentre=self.phasecentre,
                                             weight=1.0, polarisation_frame=PolarisationFrame('stokesI'))
        self.residualvis.data['uvw'][:, 2] = 0.0
        self.residualvis.data['vis'] = self.modelvis.data['vis'] - self.componentvis.data['vis']
        
        self._checkdirty(self.residualvis, 'test_predict_2d_residual', fluxthreshold=0.2)
    
    def _predict_base(self, predict, fluxthreshold=1.0):
        self.modelvis = create_visibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                          weight=1.0, polarisation_frame=PolarisationFrame('stokesI'))
        self.modelvis.data['vis'] *= 0.0
        predict(self.modelvis, self.model, **self.params)
        self.residualvis = create_visibility(self.lowcore, self.times, self.frequency,
                                             channel_bandwidth=self.channel_bandwidth,
                                             phasecentre=self.phasecentre,
                                             weight=1.0, polarisation_frame=PolarisationFrame('stokesI'))
        self.residualvis.data['uvw'][:, 2] = 0.0
        self.residualvis.data['vis'] = self.modelvis.data['vis'] - self.componentvis.data['vis']
        self._checkdirty(self.residualvis, 'test_%s_residual' % predict.__name__, fluxthreshold=fluxthreshold)
    
    def test_predict_facets(self):
        self.actualSetUp()
        self._predict_base(predict_facets, fluxthreshold=1e-7)

    def test_predict_timeslice(self):
        # This works very poorly because of the poor interpolation accuracy for point sources
        self.actualSetUp()
        for self.params['nprocessor'] in [1, 4]:
            self._predict_base(predict_timeslice, fluxthreshold=10.0)

    def test_predict_wslice(self):
        self.actualSetUp()
        self.params['wslice']=10.0
        self.params['imaginary'] = True
        self.actualSetUp()
        self._predict_base(predict_wslice, fluxthreshold=2.0)

    def test_predict_wprojection(self):
        self.actualSetUp()
        self.params = {'npixel': 256,
                       'npol': 1,
                       'cellsize': 0.001,
                       'padding': 2,
                       'oversampling': 4,
                       'wstep': 10.0}
        
        self._predict_base(predict_wprojection, fluxthreshold=2.0)
    
    def test_invert_2d(self):
        """Test if the 2D invert works with w set to zero

        Set w=0 so that the two-dimensional transform should agree exactly with the model.
        Good check on the grid correction in the vis->image direction
        """
        # Set all w to zero
        self.actualSetUp()
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency,
                                              channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                              weight=1.0, polarisation_frame=PolarisationFrame('stokesI'))
        self.componentvis.data['uvw'][:, 2] = 0.0
        self.componentvis.data['vis'] *= 0.0
        # Predict the visibility using direct evaluation
        for comp in self.components:
            predict_skycomponent_visibility(self.componentvis, comp)
        
        dirty2d = create_empty_image_like(self.model)
        dirty2d, sumwt = invert_2d(self.componentvis, dirty2d, **self.params)
        
        dirty2d = normalize_sumwt(dirty2d, sumwt)
        
        export_image_to_fits(dirty2d, '%s/test_invert_2d_dirty.fits' % self.dir)
        
        self._checkcomponents(dirty2d, fluxthreshold=20.0, positionthreshold=1.0)
    
    def _invert_base(self, invert, fluxthreshold=20.0, positionthreshold=1.0):
        dirtyFacet = create_empty_image_like(self.model)
        dirtyFacet, sumwt = invert(self.componentvis, dirtyFacet, **self.params)
        assert sumwt.all() > 0.0
        dirtyFacet = normalize_sumwt(dirtyFacet, sumwt)
        export_image_to_fits(dirtyFacet, '%s/test_%s_dirty.fits' % (self.dir, invert.__name__))
        self._checkcomponents(dirtyFacet, fluxthreshold, positionthreshold)
    
    def test_invert_facets(self):
        self.actualSetUp()
        self._invert_base(invert_facets, positionthreshold=1.0)

    def test_invert_wslice(self):
        self.actualSetUp()
        self.params = {'npixel': 256,
                       'cellsize': 0.001,
                       'padding': 2,
                       'oversampling': 4,
                       'wslice': 1.0,
                       'imaginary': True}
        for self.params['nprocessor'] in [1, 4]:
            self.actualSetUp()
            self._invert_base(invert_wslice, positionthreshold=8.0)

    def test_invert_timeslice(self):
        self.actualSetUp()
        for self.params['nprocessor'] in [1, 4]:
            self.actualSetUp()
            self._invert_base(invert_timeslice, positionthreshold=8.0)

    def test_invert_wprojection(self):
        self.actualSetUp()
        self.params = {'npixel': 256,
                      'cellsize': 0.001,
                       'padding': 2,
                       'oversampling': 4,
                       'wstep': 10.0}
        
        self._invert_base(invert_wprojection, positionthreshold=1.0)

    def test_weighting(self):
        self.actualSetUp()
        vis, density, densitygrid = weight_visibility(self.componentvis, self.model, weighting='uniform')
        assert vis.nvis == self.componentvis.nvis
        assert len(density) == vis.nvis
        assert numpy.std(vis.imaging_weight) > 0.0
        assert densitygrid.data.shape == self.model.data.shape
        vis, density, densitygrid = weight_visibility(self.componentvis, self.model, weighting='natural')
        assert density is None
        assert densitygrid is None
        
    def test_create_image_from_visibility(self):
        self.actualSetUp()
        im = create_image_from_visibility(self.componentvis, nchan=1, npixel=128)
        assert im.data.shape == (1, 1, 128, 128)

    def test_create_image_from_blockvisibility(self):
        self.actualSetUp()
        self.componentvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                                   channel_bandwidth=self.channel_bandwidth,
                                                   phasecentre=self.phasecentre, weight=1.0,
                                                   polarisation_frame=PolarisationFrame('stokesI'))
        im = create_image_from_visibility(self.componentvis, nchan=1, npixel=128)
        assert im.data.shape == (1, 1, 128, 128)
        im = create_image_from_visibility(self.componentvis, frequency=self.frequency, npixel=128)
        assert im.data.shape == (len(self.frequency), 1, 128, 128)
        im = create_image_from_visibility(self.componentvis, frequency=self.frequency, npixel=128,
                                          nchan=1)
        assert im.data.shape == (1, 1, 128, 128)

    def test_create_w_term_image(self):
        self.actualSetUp()
        im = create_w_term_image(self.componentvis, nchan=1, npixel=128)
        assert im.data.dtype == 'complex128'
        assert im.data.shape == (128, 128)
        self.assertAlmostEqual(numpy.max(im.data.real), 1.0, 7)
        im = create_w_term_image(self.componentvis, w=10.0, nchan=1, npixel=128)
        assert im.data.shape == (128, 128)
        assert im.data.dtype == 'complex128'
        self.assertAlmostEqual(numpy.max(im.data.real), 1.0, 7)
        


if __name__ == '__main__':
    unittest.main()
