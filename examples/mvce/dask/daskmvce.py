import os

from dask import delayed
from distributed import Client, LocalCluster

import numpy


def get_dask_Client(timeout=30, n_workers=None, threads_per_worker=1, processes=True, create_cluster=True):
    """ Get a Dask.distributed Client for the scheduler defined externally, otherwise create

    The environment variable ARL_DASK_SCHEDULER is interpreted as pointing to the scheduler.
    and a client using that scheduler is returned. Otherwise a client is created

    :return: Dask client
    """
    scheduler = os.getenv('ARL_DASK_SCHEDULER', None)
    if scheduler is not None:
        print("Creating Dask Client using externally defined scheduler")
        c = Client(scheduler, timeout=timeout)
    
    elif create_cluster:
        if n_workers is not None:
            cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes)
        else:
            cluster = LocalCluster(threads_per_worker=threads_per_worker, processes=processes)
        print("Creating LocalCluster and Dask Client")
        c = Client(cluster)

    else:
        c = Client()
    
    print(c)
    
    addr = c.scheduler_info()['address']
    services = c.scheduler_info()['services']
    if 'bokeh' in services.keys():
        bokeh_addr = 'http:%s:%s' % (addr.split(':')[1], services['bokeh'])
        print('Diagnostic pages available on port %s' % bokeh_addr)
    return c

def init_sparse(n, margin=0.1):
    numpy.random.seed(8753193)
    return numpy.array([numpy.random.uniform(margin, 1.0-margin, n),
                        numpy.random.uniform(margin, 1.0-margin, n)]).reshape([n,2])

def make_psf(sparse_data, shape=(512,512), footprint=3):
    grid = numpy.zeros(shape, dtype='complex')
    for i, spot in enumerate(sparse_data):
        loc = numpy.round(shape * spot).astype('int')
        grid[loc[1]-footprint//2,loc[0]+footprint//2] += 1.0
    return numpy.fft.fft(grid).real

def accumulate(psf_list):
    psf = 0.0*psf_list[0]
    for p in psf_list:
        psf += p
    return psf

if __name__ == '__main__':
    import sys
    len_chunk = 1000000
    nchunks=100
    
    sparse_graph_list=[delayed(init_sparse)(len_chunk) for i in range(nchunks)]
    psf_graph_list=[delayed(make_psf)(s) for s in sparse_graph_list]
    psf_graph = delayed(accumulate)(psf_graph_list)
    psf_graph.visualize('psf_graph.png')
    
    client= get_dask_Client(n_workers=int(sys.argv[1]))
    psf = client.compute(psf_graph, sync=True)
    client.shutdown()

    print("***** Successfully reached end *****")
    print(numpy.max(psf))
    exit()




