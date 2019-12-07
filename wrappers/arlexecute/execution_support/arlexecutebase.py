""" Execute wrap dask such that with the same code Dask.delayed can be replaced by immediate calculation

"""

__all__ = ['ARLExecuteBase', 'arlexecute']

import logging
import time

from dask import delayed, optimize
from dask.distributed import Client, wait

# Support daliuge's delayed function, make it fail if not available but used
try:
    from dlg import delayed as dlg_delayed
    from dlg.dask_emulation import compute as dlg_compute
except ImportError:
    def dlg_delayed(*args, **kwargs):
        raise Exception("daliuge is not available")

log = logging.getLogger(__name__)


class _ARLExecuteBase():
    
    _instance = None
    
    def __init__(self, use_dask=True, use_dlg=False, verbose=False, optimize=True):
        if bool(use_dask) and bool(use_dlg):
            raise ValueError('use_dask and use_dlg cannot be specified together')
        self._set_state(use_dask, use_dlg, None, verbose, optimize)

    def _set_state(self, use_dask, use_dlg, client, verbose, optimize):
        self._using_dask = use_dask
        self._using_dlg = use_dlg
        self._client = client
        self._verbose = verbose
        self._optimize = optimize

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
    
    def set_client(self, client=None, use_dask=True, use_dlg=False, verbose=False, optim=True, **kwargs):
        """Set the Dask/DALiuGE client to be used
        
        !!!This must be called before calling execute!!!!
        
        If you want to customise the Client or use an externally defined Scheduler use get_dask_Client and pass it in.
        
        :param use_dask: Use Dask?
        :param client: If None and use_dask is True, a client will be created otherwise the client is None
        :param use_dlg: Use Daliuge to execute graphs?
        :param verbose: Be verbose in output
        :param optim: Use dask.optimize via arlexecute.optimize function.
        :return:
        """
        if bool(use_dask) and bool(use_dlg):
            raise ValueError('use_dask and use_dlg cannot be specified together')

        if isinstance(self._client, Client):
            print("Removing existing client")
            self.client.close()

        if use_dask:
            client = client or Client(**kwargs)
            assert isinstance(client, Client)
            self._set_state(True, False, client, verbose, optim)
            self._client.profile()
            self._client.get_task_stream()
            self.start_time = time.time()

        elif use_dlg:
            self._set_state(False, True, client, verbose, optim)
        else:
            self._set_state(False, False, None, verbose, optim)
        if self._verbose:
            print('arlexecute.set_client: defined Dask Client')


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
                if self._verbose:
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
        
    def optimize(self, *args, **kwargs):
        """ Run optimisation of graphs
        
        Only does something when using dask
        
        :param args:
        :param kwargs:
        :return:
        """
        if self.using_dask and self._optimize:
            return optimize(*args, **kwargs)[0]
        else:
            return args[0]

    
    @property
    def client(self):
        return self._client
    
    @property
    def using_dask(self):
        return self._using_dask
    
    @property
    def using_dlg(self):
        return self._using_dlg

    @property
    def optimizing(self):
        return self._optimize

    def close(self):
        if self._using_dask and isinstance(self._client, Client):
            if self._verbose:
                print('arlexcute.close: closed down Dask Client')
            if self._client.cluster is not None:
                self._client.cluster.close()
            self._client.close()
            self._client = None

    def init_statistics(self):
        """
        Initialise the profile and task stream info
        :return:
        """
        self.start_time = time.time()
        if self._using_dask:
            self._client.profile()
            self._client.get_task_stream()

    def save_statistics(self, name='dask'):
        
        if self._using_dask:
            task_stream, graph = self.client.get_task_stream(plot='save',
                                                             filename="%s_task_stream.html" % name)
            self.client.profile(plot='save', filename="%s_profile.html" % name)
        
            def print_ts(ts):
                print(">>> Processor time used in each function")
                summary = {}
                number = {}
                for t in ts:
                    name = t['key'].split('-')[0]
                    elapsed = t['startstops'][0][2] - t['startstops'][0][1]
                    if name not in summary.keys():
                        summary[name] = elapsed
                        number[name] = 1
                    else:
                        summary[name] += elapsed
                        number[name] += 1
                total = 0.0
                for key in summary.keys():
                    total += summary[key]
                for key in summary.keys():
                    print(">>> %s %.3f (s) %.1f %s %d (calls)" %
                          (key, summary[key], 100.0 * summary[key] / total, '%', number[key]))
                print(">>> Total processor time %.3f (s)" % total)
                duration = time.time() - self.start_time
                print(">>> Total wallclock time %.3f (s)" % duration)
                speedup = (total / duration)
                print(">>> Speedup = %.2f" % speedup)
                
            print_ts(task_stream)

def ARLExecuteBase(*args, **kwargs):
    if _ARLExecuteBase._instance is None:
        _ARLExecuteBase._instance = _ARLExecuteBase(*args, **kwargs)
    return _ARLExecuteBase._instance

# Any new arlexecute created by import of this file points to the only _ARLExecuteBase
arlexecute = ARLExecuteBase(use_dask=True)