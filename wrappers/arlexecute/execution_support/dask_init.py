""" Initialise dask


"""

import logging
import os

from distributed import Client, LocalCluster

log = logging.getLogger(__name__)


def get_dask_Client(timeout=30, n_workers=None, threads_per_worker=1, processes=True, create_cluster=True,
                    memory_limit=None, local_dir='.', with_file=False,
                    scheduler_file='./scheduler.json',
                    dashboard_address=':8787'):
    """ Get a Dask.distributed Client for the scheduler defined externally, otherwise create

    The environment variable ARL_DASK_SCHEDULER is interpreted as pointing to the scheduler.
    and a client using that scheduler is returned. Otherwise a client is created

    :param timeout: Time out for creation
    :param n_workers: Number of workers
    :param threads_per_worker:
    :param processes: Use processes instead of threads
    :param create_cluster: Create a LocalCluster
    :param memory_limit: Memory limit per worker (bytes e.g. 8e9)
    :return: Dask client
    """
    scheduler = os.getenv('ARL_DASK_SCHEDULER', None)
    if scheduler is not None:
        print("Creating Dask Client using externally defined scheduler")
        c = Client(scheduler, timeout=timeout)
    elif with_file:
        print("Creating Dask Client using externally defined scheduler in file  %s" % scheduler_file)
        c = Client(scheduler_file=scheduler_file, timeout=timeout)
        
    elif create_cluster:
        if n_workers is not None:
            if memory_limit is not None:
                cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes,
                                       memory_limit=memory_limit, local_dir=local_dir,
                                       dashboard_address=dashboard_address)
            else:
                cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes,
                                       local_dir=local_dir,
                                       dashboard_address=dashboard_address)
        else:
            if memory_limit is not None:
                cluster = LocalCluster(threads_per_worker=threads_per_worker, processes=processes,
                                       memory_limit=memory_limit, local_dir=local_dir,
                                       dashboard_address=dashboard_address)
            else:
                cluster = LocalCluster(threads_per_worker=threads_per_worker, processes=processes, local_dir=local_dir,
                                       dashboard_address=dashboard_address)
        
        print("Creating LocalCluster and Dask Client")
        c = Client(cluster)
    else:
        c = Client(threads_per_worker=threads_per_worker, processes=processes,
                   memory_limit=memory_limit, local_dir=local_dir)
    
    addr = c.scheduler_info()['address']
    services = c.scheduler_info()['services']
    if 'bokeh' in services.keys():
        bokeh_addr = 'http:%s:%s' % (addr.split(':')[1], services['bokeh'])
        print('Diagnostic pages available on port %s' % bokeh_addr)
    return c


def get_nodes():
    """ Get the nodes being used

    The environment variable ARL_HOSTFILE is interpreted as file containing the nodes

    :return: List of strings
    """
    hostfile = os.getenv('ARL_HOSTFILE', None)
    if hostfile is None:
        print("No hostfile specified")
        return None
    
    import socket
    with open(hostfile, 'r') as file:
        nodes = [line.replace('\n', '') for line in file.readlines()]
        print("Nodes being used are %s" % nodes)
        nodes = [socket.gethostbyname(node) for node in nodes]
        print("Nodes IPs are %s" % nodes)
        return nodes


def findNodes(c):
    """ Find Nodes being used for this Client
    
    """
    return [c.scheduler_info()['workers'][name]['host'] for name in c.scheduler_info()['workers'].keys()]
