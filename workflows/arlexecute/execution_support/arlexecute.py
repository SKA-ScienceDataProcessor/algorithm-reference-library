""" Execute wrap dask such that with the same code Dask.delayed can be replaced by immediate calculation

"""

import logging
import time

from dask import delayed
from dask.distributed import Client, wait

log = logging.getLogger(__name__)


class ARLExecuteBase():
    
    def __init__(self, use_dask=True):
        self._using_dask = use_dask
        self._client = None
    
    def execute(self, func, *args, **kwargs):
        """ Wrap for immediate or deferred execution
        
        Passes through if dask is not being used
        
        :param args:
        :param kwargs:
        :return: delayed func or func
        """
        if self._using_dask:
            return delayed(func, *args, **kwargs)
        else:
            return func
    
    def type(self):
        """ Get the type of the execution system
        
        :return:
        """
        if self._using_dask:
            return 'dask'
        else:
            return 'function'
    
    def set_client(self, client=None, use_dask=True, **kwargs):
        """Set the Dask client to be used
        
        !!!This must be called before calling execute!!!!
        
        :param use_dask: Use Dask?
        :param client: If None and use_dask is True, a client will be created otherwise the client is None
        :return:
        """
        if isinstance(self._client, Client):
            self.client.close()
        
        if use_dask:
            if client is None:
                self._client = Client(**kwargs)
            else:
                assert isinstance(client, Client)
                self._client = client
            self._using_dask = True
        else:
            self._client = None
            self._using_dask = False
    
    def compute(self, value, sync=False):
        """Get the actual value
        
        If not using dask then this returns the value directly since it already is computed
        If using dask and sync=True then this waits and resturns the actual wait.
        If using dask and sync=False then this returns a future, on which you will need to call .result()
        
        :param value:
        :return:
        """
        if self._using_dask:
            start = time.time()
            if self.client is None:
                return value.compute()
            else:
                future = self.client.compute(value, sync=sync)
                wait(future)
                duration = time.time() - start
                log.debug("arlexecute.compute: Execution using Dask took %.3f seconds" % duration)
                print("arlexecute.compute: Execution using Dask took %.3f seconds" % duration)
            return future
        else:
            return value
    
    def scatter(self, graph):
        """Scatter graph to workers

        No-op if using_dask is False
        :param graph:
        :return:
        """
        if self.using_dask and self.client is not None:
            return self.client.scatter(graph)
        else:
            return graph
    
    def gather(self, graph):
        """Gather graph from workers

        No-op if using_dask is False
        :param graph:
        :return:
        """
        if self.using_dask and self.client is not None:
            return self.client.gather(graph)
        else:
            return graph
    
    def run(self, func, *args, **kwargs):
        """ Run a function on the client
        
        :param func:
        :return:
        """
        if self.using_dask:
            return self.client.run(func, *args, **kwargs)
        else:
            return func
    
    @property
    def client(self):
        return self._client
    
    @property
    def using_dask(self):
        return self._using_dask
    
    def close(self):
        if self._using_dask and isinstance(self._client, Client):
            print('arlexcute.close: closing down Dask Client')
            self._client.close()


arlexecute = ARLExecuteBase(use_dask=True)
