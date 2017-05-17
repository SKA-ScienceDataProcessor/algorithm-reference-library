""" Initialise dask"""

import os
import logging

log = logging.getLogger(__name__)

from distributed import Client

def get_dask_Client(timeout=30):
    """ Get a dask Client. the dask scheduler defined externally, otherwise create
    
    :return:
    """
    scheduler = os.getenv('ARL_DASK_SCHEDULER', None)
    if scheduler is not None:
        print("Creating Dask Client using externally defined scheduler")
        return Client(scheduler, timeout=timeout)
    else:
        print("Creating Dask Client")
        return Client()
