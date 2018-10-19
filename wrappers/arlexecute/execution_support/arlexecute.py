""" Execute wrap dask such that with the same code Dask.delayed can be replaced by immediate calculation

"""

import logging
import time

from dask import delayed
from dask.distributed import Client, wait

# Support daliuge's delayed function, make it fail if not available but used
try:
    from dlg import delayed as dlg_delayed
    from dlg.dask_emulation import compute as dlg_compute
except ImportError:
    def dlg_delayed(*args, **kwargs):
        raise Exception("daliuge is not available")

log = logging.getLogger(__name__)


class ARLExecuteBase():
    
    def __init__(self, use_dask=True, use_dlg=False):
        if bool(use_dask) and bool(use_dlg):
            raise ValueError('use_dask and use_dlg cannot be specified together')
        self._set_state(use_dask, use_dlg, None)

    def _set_state(self, use_dask, use_dlg, client):
        self._using_dask = use_dask
        self._using_dlg = use_dlg
        self._client = client

    def execute(self, func, *args, **kwargs):
        """ Wrap for immediate or deferred execution
        
        Passes through if dask is not being used
        
        :param args:
        :param kwargs:
        :return: delayed func or func
        """
        if self._using_dask:
            return delayed(func, *args, **kwargs)
        elif self._using_dlg:
            return dlg_delayed(func, *args, **kwargs)
        else:
            return func
    
    def type(self):
        """ Get the type of the execution system
        
        :return:
        """
        if self._using_dask:
            return 'dask'
        elif self._using_dlg:
            return 'daliuge'
        else:
            return 'function'
    
    def set_client(self, client=None, use_dask=True, use_dlg=False, **kwargs):
        """Set the Dask/DALiuGE client to be used
        
        !!!This must be called before calling execute!!!!
        
        If you want to customise the Client or use an externally defined Scheduler use get_dask_Client and pass it in.
        
        :param use_dask: Use Dask?
        :param client: If None and use_dask is True, a client will be created otherwise the client is None
        :return:
        """
        if bool(use_dask) and bool(use_dlg):
            raise ValueError('use_dask and use_dlg cannot be specified together')

        if isinstance(self._client, Client):
            self.client.close()
        
        if use_dask:
            client = client or Client(**kwargs)
            assert isinstance(client, Client)
            self._set_state(True, False, client)
        elif use_dlg:
            self._set_state(False, True, client)
        else:
            self._set_state(False, False, None)
    
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
        elif self._using_dlg:
            kwargs = {'client': self._client} if self._client else {}
            return dlg_compute(value, **kwargs)
        else:
            return value

    def persist(self, graph, **kwargs):
        """Persist graph data on workers

        No-op if using_dask is False
        :param graph:
        :return:
        """
        if self.using_dask and self.client is not None:
            return self.client.persist(graph, **kwargs)
        else:
            return graph

    def scatter(self, graph, **kwargs):
        """Scatter graph data to workers

        No-op if using_dask is False
        :param graph:
        :return:
        """
        if self.using_dask and self.client is not None:
            return self.client.scatter(graph, **kwargs)
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
    
    @property
    def using_dlg(self):
        return self._using_dlg

    def close(self):
        if self._using_dask and isinstance(self._client, Client):
            print('arlexcute.close: closing down Dask Client')
            try:
                self._client.cluster.close()
            except:
                pass
            try:
                self._client.close()
            except:
                pass


arlexecute = ARLExecuteBase(use_dask=True)
