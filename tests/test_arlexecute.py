""" Unit tests for testing support


"""
import logging
import unittest

import numpy

from arl.graphs.execute import arlexecute
from dask.distributed import Client, Future
from dask import delayed

log = logging.getLogger(__name__)


class TestARLExecute(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_useDaskAsync(self):
        def square(x):
            return x ** 2
        
        arlexecute.set_client(n_workers=4)
        graph = arlexecute.execute(square)(numpy.arange(10))
        result = arlexecute.compute(graph, sync=False)
        assert isinstance(result, Future)
        assert (result.result() == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])).all()
    
    def test_useFunction(self):
        def square(x):
            return x ** 2
        
        arlexecute.set_client(use_dask=False)
        graph = arlexecute.execute(square)(numpy.arange(10))
        assert (arlexecute.compute(graph) == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])).all()
    
    def test_setup_client(self):
        def square(x):
            return x ** 2
        
        for use_dask in [False, True, False, True, True]:
            arlexecute.set_client(use_dask=use_dask)
            graph = arlexecute.execute(square)(numpy.arange(10))
            assert (arlexecute.compute(graph) == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])).all()
    
    def test_setup_errors(self):

        with self.assertRaises(AssertionError):
            arlexecute.set_client(client=1)
            
    def test_pure_dask(self):
    
        def square(x):
            return x ** 2
    
        client = Client()
        graph = delayed(square)(numpy.arange(10))
        assert (client.compute(graph, sync=True) == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])).all()
