""" Unit tests for testing support


"""
import logging
import unittest

import numpy

# Import the base and then make a global version
from arl.wrappers.arlexecute.execution_support import ARLExecuteBase

log = logging.getLogger(__name__)

class TestARLExecute(unittest.TestCase):
    
    def setUp(self):
        global arlexecute
        arlexecute = ARLExecuteBase(use_dask=True)
        arlexecute.set_client(use_dask=True, verbose=False)
        
    def tearDown(self):
        arlexecute.close()
    
    def test_useFunction(self):
        def square(x):
            return x ** 2

        graph = arlexecute.execute(square)(numpy.arange(10))
        result = arlexecute.compute(graph, sync=True)
        assert (result == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])).all(), result

    def test_useDaskAsync(self):
        def square(x):
            return x ** 2
    
        graph = arlexecute.execute(square)(numpy.arange(10))
        result = arlexecute.compute(graph).result()
        assert (result == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])).all()

    def test_useDaskSync(self):
        def square(x):
            return x ** 2
    
        graph = arlexecute.execute(square)(numpy.arange(10))
        result = arlexecute.compute(graph, sync=True)
        assert (result == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])).all()
