"""Unit tests for pipelines expressed via dask.delayed


"""

import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from workflows.serial.calibration.calibration_serial import calibrate_list_serial_workflow
from processing_components.calibration.calibration_control import create_calibration_controls
from processing_components.calibration.operations import create_gaintable_from_blockvisibility, apply_gaintable
from processing_components.simulation.testing_support import ingest_unittest_visibility
from processing_components.simulation.configurations import create_named_configuration
from processing_components.simulation.testing_support import simulate_gaintable
from processing_components.visibility.base import copy_visibility

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestCalibrateGraphs(unittest.TestCase):
    
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.persist = False
    
    def tearDown(self):
        pass
    
    def actualSetUp(self, nfreqwin=3, dospectral=True, dopol=False,
                    amp_errors=None, phase_errors=None, zerow=True):
        
        if amp_errors is None:
            amp_errors = {'T': 0.0, 'G': 0.1}
        if phase_errors is None:
            phase_errors = {'T': 1.0, 'G': 0.0}
        
        self.npixel = 512
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = nfreqwin
        self.vis_list = list()
        self.ntimes = 1
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        
        if self.freqwin > 1:
            self.channelwidth = numpy.array(self.freqwin * [self.frequency[1] - self.frequency[0]])
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
        self.blockvis_list = [ingest_unittest_visibility(self.low,
                                                         [self.frequency[i]],
                                                         [self.channelwidth[i]],
                                                         self.times,
                                                         self.vis_pol,
                                                         self.phasecentre, block=True,
                                                         zerow=zerow)
                              for i in range(nfreqwin)]
        
        for v in self.blockvis_list:
            v.data['vis'][...] = 1.0 + 0.0j
        
        self.error_blockvis_list = [copy_visibility(v) for v in self.blockvis_list]
        gt = create_gaintable_from_blockvisibility(self.blockvis_list[0])
        gt = simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.0, smooth_channels=1, leakage=0.0)
        self.error_blockvis_list = [apply_gaintable(self.error_blockvis_list[i], gt)
                                    for i in range(self.freqwin)]
        
        assert numpy.max(numpy.abs(self.error_blockvis_list[0].vis - self.blockvis_list[0].vis)) > 0.0
    
    def test_time_setup(self):
        self.actualSetUp()
    
    def test_calibrate_serial(self):
        amp_errors = {'T': 0.0, 'G': 0.0}
        phase_errors = {'T': 1.0, 'G': 0.0}
        self.actualSetUp(amp_errors=amp_errors, phase_errors=phase_errors)
        
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        controls['T']['timeslice'] = 'auto'
        
        calibrate_list = \
            calibrate_list_serial_workflow(self.error_blockvis_list, self.blockvis_list,
                                           calibration_context='T', controls=controls, do_selfcal=True,
                                           global_solution=False)
        assert len(calibrate_list) == 2
        assert numpy.max(calibrate_list[1][0]['T'].residual) < 7e-6, numpy.max(calibrate_list[1][0]['T'].residual)
        assert numpy.max(numpy.abs(calibrate_list[0][0].vis - self.blockvis_list[0].vis)) < 2e-6
    
    def test_calibrate_serial_empty(self):
        amp_errors = {'T': 0.0, 'G': 0.0}
        phase_errors = {'T': 1.0, 'G': 0.0}
        self.actualSetUp(amp_errors=amp_errors, phase_errors=phase_errors)
        
        for v in self.blockvis_list:
            v.data['vis'][...] = 0.0 + 0.0j
        
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        controls['T']['timeslice'] = 'auto'
        
        calibrate_list = \
            calibrate_list_serial_workflow(self.error_blockvis_list, self.blockvis_list,
                                                calibration_context='T', controls=controls, do_selfcal=True,
                                                global_solution=False)
        assert len(calibrate_list[1][0]) == 1
        assert numpy.max(calibrate_list[1][0]['T'].residual) == 0.0, numpy.max(calibrate_list[1][0]['T'].residual)


    def test_calibrate_serial_global(self):
        amp_errors = {'T': 0.0, 'G': 0.0}
        phase_errors = {'T': 1.0, 'G': 0.0}
        self.actualSetUp(amp_errors=amp_errors, phase_errors=phase_errors)
        
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        controls['T']['timeslice'] = 'auto'
        
        calibrate_list = \
            calibrate_list_serial_workflow(self.error_blockvis_list, self.blockvis_list,
                                           calibration_context='T', controls=controls, do_selfcal=True,
                                           global_solution=True)
        
        assert len(calibrate_list) == 2
        assert numpy.max(calibrate_list[1][0]['T'].residual) < 7e-6, numpy.max(calibrate_list[1][0]['T'].residual)
        err = numpy.max(numpy.abs(calibrate_list[0][0].vis - self.blockvis_list[0].vis))
        assert err < 2e-6, err
    
    def test_calibrate_serial_global_empty(self):
        amp_errors = {'T': 0.0, 'G': 0.0}
        phase_errors = {'T': 1.0, 'G': 0.0}
        self.actualSetUp(amp_errors=amp_errors, phase_errors=phase_errors)
        
        for v in self.blockvis_list:
            v.data['vis'][...] = 0.0 + 0.0j
        
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        controls['T']['timeslice'] = 'auto'
        
        calibrate_list = \
            calibrate_list_serial_workflow(self.error_blockvis_list, self.blockvis_list,
                                                    calibration_context='T', controls=controls, do_selfcal=True,
                                                    global_solution=True)
        assert len(calibrate_list[1][0]) == 1
        assert numpy.max(calibrate_list[1][0]['T'].residual) == 0.0, numpy.max(calibrate_list[1][0]['T'].residual)


if __name__ == '__main__':
    unittest.main()
