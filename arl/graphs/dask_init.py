""" Initialise dask


"""

import os
import logging

log = logging.getLogger(__name__)

from distributed import Client, LocalCluster

def get_dask_Client(timeout=30, n_workers=None, threads_per_worker=1, processes=True):
    """ Get a Dask.distributed Client for the scheduler defined externally, otherwise create
    
    The environment variable ARL_DASK_SCHEDULER is interpreted as pointing to the scheduler.
    and a client using that scheduler is returned. Otherwise a client is created
    
    :return: Dask client
    """
    scheduler = os.getenv('ARL_DASK_SCHEDULER', None)
    if scheduler is not None:
        print("Creating Dask Client using externally defined scheduler")
        c = Client(scheduler, timeout=timeout)
        print(c)
    else:
        if n_workers is not None:
            cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes)
        else:
            cluster = LocalCluster(threads_per_worker=threads_per_worker, processes=processes)
        print("Creating LocalCluster and Dask Client")
        c = Client(cluster)
        print(c)
        
    addr = c.scheduler_info()['address']
    services = c.scheduler_info()['services']
    if 'bokeh' in services.keys():
        bokeh_addr = 'http:%s:%s' % (addr.split(':')[1], services['bokeh'])
        print('Diagnostic pages available on port %s' % bokeh_addr)
    return c


def kill_dask_Scheduler(client):
    """ Kill the process dask-ssh
    
    :params c: Dask client
    
    """
    import psutil, signal
    for proc in psutil.process_iter():
        # check whether the process name matches
        if proc.name() == "graphs-ssh":
            proc.send_signal(signal.SIGHUP)
            
def kill_dask_Client(c):
    """ Kill the Client
    
    :params c: Dask client
    """
    c.loop.add_callback(c.scheduler.retire_workers, close_workers=True)
    c.loop.add_callback(c.scheduler.terminate)
    c.run_on_scheduler(lambda dask_scheduler: dask_scheduler.loop.stop())
    c.shutdown()

