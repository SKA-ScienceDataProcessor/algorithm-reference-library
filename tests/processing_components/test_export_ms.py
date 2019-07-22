# -*- coding: utf-8 -*-

"""Unit test for the measurementset module."""

import os
import time
import unittest
import tempfile
import numpy
import shutil

import logging
import sys
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from processing_components.simulation.configurations import create_named_configuration
from processing_components.visibility.base import create_visibility
from processing_components.visibility import msv2
from data_models.memory_data_models import Visibility, BlockVisibility, Configuration
from data_models.polarisation import PolarisationFrame, ReceptorFrame, correlate_polarisation


run_ms_tests = False
try:
    import casacore
    run_ms_tests = True
except ImportError:
    pass

@unittest.skipUnless(run_ms_tests, "requires the 'casacore' module")
class measurementset_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the lsl.writer.measurementset.Ms
    class."""
    
    testPath = None
    
    def setUp(self):
        """Turn off all numpy warnings and create the temporary file directory."""
       
        numpy.seterr(all='ignore')
        self.testPath = tempfile.mkdtemp(prefix='test-measurementset-', suffix='.tmp')
        
    def __initData(self):
        """Private function to generate a random set of data for writing a UVFITS
        file.  The data is returned as a dictionary with keys:
         * freq - frequency array in Hz
         * site - lwa.common.stations object
         * stands - array of stand numbers
         * bl - list of baseline pairs in real stand numbers
         * vis - array of visibility data in baseline x freq format
        """

        # Frequency range
        freq = numpy.arange(0,512)*20e6/512 + 40e6
        # Site and stands
        mount = numpy.array(['equat','equat','equat','equat','equat','equat','equat','equat','equat','equat'])
        names = numpy.array(['ak02','ak04','ak05','ak12','ak13','ak14','ak16','ak24','ak28','ak30'])
        diameter = numpy.array([12.,12.,12.,12.,12.,12.,12.,12.,12.,12.])
        xyz = numpy.array([[-2556109.98244348,5097388.70050131,-2848440.1332423 ],
            [-2556087.396082  ,  5097423.589662  , -2848396.867933  ],
            [-2556028.60254059,  5097451.46195695, -2848399.83113161],
            [-2556496.23893101,  5097333.71466669, -2848187.33832738],
            [-2556407.35299627,  5097064.98390756, -2848756.02069474],
            [-2555972.78456557,  5097233.65481756, -2848839.88915184],
            [-2555592.88867802,  5097835.02121109, -2848098.26409648],
            [-2555959.34313275,  5096979.52802882, -2849303.57702486],
            [-2556552.97431815,  5097767.23612874, -2847354.29540396],
            [-2557348.40370367,  5097170.17682775, -2847716.21368966]])
        site_config = Configuration(name='', data=None, location=None,
                                        names=names, xyz=xyz, mount=mount, frame=None,
                                        receptor_frame=ReceptorFrame("linear"),
                                        diameter=diameter)
        antennas = list(range(10))

        # Set baselines and data
        blList = []
        N = len(antennas)

        antennas2 = antennas

        for i in range(0, N - 1):
            for j in range(i + 1, N):
                blList.append((antennas[i], antennas2[j]))

        visData = numpy.random.rand(len(blList), len(freq))
        visData = visData.astype(numpy.complex64)
        
        return {'freq': freq, 'site': site_config, 'antennas': antennas, 'bl': blList, 'vis': visData}
        
    def test_write_tables(self):
        """Test if the MeasurementSet writer writes all of the tables."""
        
        testTime = time.time()
        testFile = os.path.join(self.testPath, 'ms-test-W.ms')
        
        # Get some data
        data = self.__initData()
        
        # Start the table
        tbl = msv2.Ms(testFile, ref_time=testTime)
        tbl.set_stokes(['xx'])
        tbl.set_frequency(data['freq'])
        tbl.set_geometry(data['site'], data['antennas'])
        tbl.add_data_set(testTime, 6.0, data['bl'], data['vis'])
        tbl.write()
        
        # Make sure everyone is there
        self.assertTrue(os.path.exists(testFile))
        for tbl in ('ANTENNA', 'DATA_DESCRIPTION', 'FEED', 'FIELD', 'FLAG_CMD', 'HISTORY', 
                    'OBSERVATION', 'POINTING', 'POLARIZATION', 'PROCESSOR', 'SOURCE', 
                    'SPECTRAL_WINDOW', 'STATE'):
            self.assertTrue(os.path.exists(os.path.join(testFile, tbl)))
            
    def test_main_table(self):
        """Test the primary data table."""
        
        testTime = time.time()
        testFile = os.path.join(self.testPath, 'ms-test-UV.ms')
        
        # Get some data
        data = self.__initData()
        
        # Start the file
        fits = msv2.Ms(testFile, ref_time=testTime)
        fits.set_stokes(['xx'])
        fits.set_frequency(data['freq'])
        fits.set_geometry(data['site'], data['antennas'])
        fits.add_data_set(testTime, 6.0, data['bl'], data['vis'])
        fits.write()
        
        # Open the table and examine
        ms = casacore.tables.table(testFile, ack=False)
        uvw  = ms.getcol('UVW')
        ant1 = ms.getcol('ANTENNA1')
        ant2 = ms.getcol('ANTENNA2')
        vis  = ms.getcol('DATA')
        
        ms2 = casacore.tables.table(os.path.join(testFile, 'ANTENNA'), ack=False)
        mapper = ms2.getcol('NAME')
        # mapper = [int(m[3:], 10) for m in mapper]
        
        # Correct number of visibilities
        self.assertEqual(uvw.shape[0], data['vis'].shape[0])
        self.assertEqual(vis.shape[0], data['vis'].shape[0])
        
        # Correct number of uvw coordinates
        self.assertEqual(uvw.shape[1], 3)
        
        # Correct number of frequencies
        self.assertEqual(vis.shape[1], data['freq'].size)
            
        # Correct values
        for row in range(uvw.shape[0]):
            stand1 = ant1[row]
            stand2 = ant2[row]
            visData = vis[row,:,0]
           
            # Find out which visibility set in the random data corresponds to the 
            # current visibility
            i = 0
            for a1,a2 in data['bl']:
                if a1 == stand1 and a2 == stand2:
                    break
                else:
                    i = i + 1
                    
            # Run the comparison
            for vd, sd in zip(visData, data['vis'][i,:]):
                self.assertAlmostEqual(vd, sd, 8)
            i = i + 1
            
        ms.close()
        ms2.close()
        
    def test_multi_if(self):
        """writing more than one spectral window to a MeasurementSet."""
        
        testTime = time.time()
        testFile = os.path.join(self.testPath, 'ms-test-MultiIF.ms')
        
        # Get some data
        data = self.__initData()
        
        # Start the file
        fits = msv2.Ms(testFile, ref_time=testTime)
        fits.set_stokes(['xx'])
        fits.set_frequency(data['freq'])
        fits.set_frequency(data['freq']+10e6)
        fits.set_geometry(data['site'], data['antennas'])
        fits.add_data_set(testTime, 6.0, data['bl'], 
                          numpy.concatenate([data['vis'], 10*data['vis']], axis=1))
        fits.write()
        
        # Open the table and examine
        ms = casacore.tables.table(testFile, ack=False)
        uvw  = ms.getcol('UVW')
        ant1 = ms.getcol('ANTENNA1')
        ant2 = ms.getcol('ANTENNA2')
        ddsc = ms.getcol('DATA_DESC_ID')
        vis  = ms.getcol('DATA')
        
        ms2 = casacore.tables.table(os.path.join(testFile, 'ANTENNA'), ack=False)
        mapper = ms2.getcol('NAME')
        # mapper = [int(m[3:], 10) for m in mapper]
        
        ms3 = casacore.tables.table(os.path.join(testFile, 'DATA_DESCRIPTION'), ack=False)
        spw = [i for i in ms3.getcol('SPECTRAL_WINDOW_ID')]
        
        # Correct number of visibilities
        self.assertEqual(uvw.shape[0], 2*data['vis'].shape[0])
        self.assertEqual(vis.shape[0], 2*data['vis'].shape[0])
        
        # Correct number of uvw coordinates
        self.assertEqual(uvw.shape[1], 3)
        
        # Correct number of frequencies
        self.assertEqual(vis.shape[1], data['freq'].size)
            
        # Correct values
        for row in range(uvw.shape[0]):
            stand1 = ant1[row]
            stand2 = ant2[row]
            descid = ddsc[row]
            visData = vis[row,:,0]
           
            # Find out which visibility set in the random data corresponds to the 
            # current visibility
            i = 0
            for a1,a2 in data['bl']:
                if a1 == stand1 and a2 == stand2:
                    break
                else:
                    i = i + 1
                    
            # Find out which spectral window this corresponds to
            if spw[descid] == 0:
                compData = data['vis']
            else:
                compData = 10*data['vis']
                
            # Run the comparison
            for vd, sd in zip(visData, compData[i,:]):
                self.assertAlmostEqual(vd, sd, 8)
                
        ms.close()
        ms2.close()
        ms3.close()
        
    def tearDown(self):
        """Remove the test path directory and its contents"""
        
        shutil.rmtree(self.testPath, ignore_errors=True)


class measurementset_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the lsl.reader units 
    tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(measurementset_tests))


if __name__ == '__main__':
    unittest.main()
