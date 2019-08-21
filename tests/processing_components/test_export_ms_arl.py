import sys
import os
sys.path.append('.')
import unittest

# dir = '/Users/wangfeng/dev/algorithm-reference-library/data/vis/ASKAP_example.ms'

import logging
import numpy

from data_models.parameters import arl_path

from processing_components.visibility.coalesce import convert_visibility_to_blockvisibility, convert_blockvisibility_to_visibility

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))

run_ms_tests = False
try:
    import casacore
    from processing_components.visibility.base import create_blockvisibility, create_blockvisibility_from_ms
    from processing_components.visibility.base import export_blockvisility_to_ms

    run_ms_tests = True
except ImportError:
    pass

@unittest.skipUnless(run_ms_tests, "requires the 'casacore' module")
class export_ms_arl_test(unittest.TestCase):

    def setUp(self):
        """Turn off all numpy warnings and create the temporary file directory."""
        pass

    def test_copy_ms(self):
        if run_ms_tests == False:
            return

        msfile = arl_path("data/vis/ASKAP_example.ms")
        msoutfile = arl_path("test_results/test_export_ms_ASKAP_output.ms")

        v = create_blockvisibility_from_ms(msfile)
        export_blockvisility_to_ms(msoutfile,v)                # vis_by_channel.append(integrate_visibility_by_channel(v[0]))

    def test_export_ms(self):
        if run_ms_tests == False:
            return

        msoutfile = arl_path("test_results/test_export_ms_ASKAP_output.ms")

        from astropy.coordinates import SkyCoord
        from astropy import units as u

        from wrappers.serial.image.operations import show_image, export_image_to_fits
        from wrappers.serial.simulation.configurations import create_named_configuration
        from wrappers.serial.simulation.testing_support import create_test_image
        from wrappers.serial.imaging.base import create_image_from_visibility
        from wrappers.serial.imaging.base import advise_wide_field

        from workflows.serial.imaging.imaging_serial import invert_list_serial_workflow, predict_list_serial_workflow

        from data_models.polarisation import PolarisationFrame

        lowr3 = create_named_configuration('LOWBD2', rmax=750.0)

        times = numpy.zeros([1])
        frequency = numpy.array([1e8])
        channelbandwidth = numpy.array([1e6])
        phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')

        bvis = create_blockvisibility(lowr3, times, frequency, phasecentre=phasecentre,
                                               weight=1.0, polarisation_frame=PolarisationFrame('stokesI'),
                                               channel_bandwidth=channelbandwidth)

        vt = convert_blockvisibility_to_visibility(bvis)

        advice = advise_wide_field(vt, guard_band_image=3.0, delA=0.1, facets=1, wprojection_planes=1,
                                   oversampling_synthesised_beam=4.0)
        cellsize = advice['cellsize']


        m31image = create_test_image(frequency=frequency, cellsize=cellsize)
        nchan, npol, ny, nx = m31image.data.shape
        m31image.wcs.wcs.crval[0] = vt.phasecentre.ra.deg
        m31image.wcs.wcs.crval[1] = vt.phasecentre.dec.deg
        m31image.wcs.wcs.crpix[0] = float(nx // 2)
        m31image.wcs.wcs.crpix[1] = float(ny // 2)
        vt = predict_list_serial_workflow([vt], [m31image], context='2d')[0]
        # uvdist = numpy.sqrt(vt.data['uvw'][:, 0] ** 2 + vt.data['uvw'][:, 1] ** 2)
        #
        # model = create_image_from_visibility(vt, cellsize=cellsize, npixel=512)
        # dirty, sumwt = invert_list_serial_workflow([vt], [model], context='2d')[0]
        # psf, sumwt = invert_list_serial_workflow([vt], [model], context='2d', dopsf=True)[0]
        #
        # show_image(dirty)
        # print("Max, min in dirty image = %.6f, %.6f, sumwt = %f" % (dirty.data.max(), dirty.data.min(), sumwt))
        #
        # print("Max, min in PSF         = %.6f, %.6f, sumwt = %f" % (psf.data.max(), psf.data.min(), sumwt))
        # results_dir="/Users/f.wang"
        # export_image_to_fits(dirty, '%s/imaging_dirty.fits' % (results_dir))
        # export_image_to_fits(psf, '%s/imaging_psf.fits' % (results_dir))

        v = convert_visibility_to_blockvisibility(vt)
        vis_list=[]
        vis_list.append(v)
        export_blockvisility_to_ms(msoutfile, vis_list,source_name='M31')

class export_measurementset_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which tests exporting measurementset
    tests."""

    def __init__(self):
        unittest.TestSuite.__init__(self)

        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(export_ms_arl_test))

if __name__ == '__main__':
    unittest.main()