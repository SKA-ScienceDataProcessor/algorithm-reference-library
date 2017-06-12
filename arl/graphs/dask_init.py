""" Initialise dask"""

import os
import logging

log = logging.getLogger(__name__)

from distributed import Client

def get_dask_Client(timeout=30):
    """ Get a graphs Client. the graphs scheduler defined externally, otherwise create
    
    :return:
    """
    scheduler = os.getenv('ARL_DASK_SCHEDULER', None)
    if scheduler is not None:
        print("Creating Dask Client using externally defined scheduler")
        return Client(scheduler, timeout=timeout)
    else:
        print("Creating Dask Client")
        return Client(timeout=timeout)
    
def kill_dask_Scheduler(client):
    """ Kill the process graphs-ssh"""
    import psutil, signal
    for proc in psutil.process_iter():
        # check whether the process name matches
        if proc.name() == "graphs-ssh":
            proc.send_signal(signal.SIGHUP)
            
def kill_dask_Client(c):
    """ Kill the Client"""
    c.loop.add_callback(c.scheduler.retire_workers, close_workers=True)
    c.loop.add_callback(c.scheduler.terminate)
    c.run_on_scheduler(lambda dask_scheduler: dask_scheduler.loop.stop())
    c.shutdown()

