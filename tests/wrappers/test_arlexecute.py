""" Unit tests for testing support


"""
import logging
import unittest

import numpy

from wrappers.arlexecute.execution_support.arlexecute import arlexecute

log = logging.getLogger(__name__)

class TestARLExecute(unittest.TestCase):
    
    def test_useFunction(self):
        def square(x):
            return x ** 2
        
        arlexecute.set_client(use_dask=False)
        graph = arlexecute.execute(square)(numpy.arange(10))
        assert (arlexecute.compute(graph) == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])).all()
        arlexecute.close()

    def test_useDaskAsync(self):
        def square(x):
            return x ** 2
    
        arlexecute.set_client(use_dask=True)
        graph = arlexecute.execute(square)(numpy.arange(10))
        result = arlexecute.compute(graph).result()
        assert (result == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])).all()
        arlexecute.close()

    def test_useDaskSync(self):
        def square(x):
            return x ** 2
    
        arlexecute.set_client(use_dask=True)
        graph = arlexecute.execute(square)(numpy.arange(10))
        result = arlexecute.compute(graph, sync=True)
        assert (result == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])).all()
        arlexecute.close()

    def test_setup_errors(self):
        with self.assertRaises(AssertionError):
            arlexecute.set_client(client=1)
