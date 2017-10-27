""" Test to illustrate losing workers under dask/distributed.

This mimics the overall structure and workload of our processing.

Tim Cornwell 9 Sept 2017
realtimcornwell@gmail.com
"""
import numpy
from dask import delayed
from distributed import Client

# Make some randomly located points on 2D plane
def init_sparse(n, margin=0.1):
    numpy.random.seed(8753193)
    return numpy.array([numpy.random.uniform(margin, 1.0 - margin, n),
                        numpy.random.uniform(margin, 1.0 - margin, n)]).reshape([n, 2])


# Put the points onto a grid and FFT
def grid_and_invert_data(sparse_data, shape):
    grid = numpy.zeros(shape, dtype='complex')
    loc = numpy.round(shape * sparse_data).astype('int')
    for i in range(0, sparse_data.shape[0]):
        grid[loc[i,:]] = 1.0
    return numpy.fft.fft(grid).real

if __name__ == '__main__':
    import sys
    import time
    
    start=time.time()
    
    # Process nchunks each of length len_chunk 2d points, making a psf of size shape
    len_chunk = 16384
    nchunks = 256*4
    nreduce = 16*4
    shape=[1024, 1024]
    skip = 1

    # We pass in the scheduler from the invoking script
    if len(sys.argv) > 1:
        scheduler = sys.argv[1]
        client = Client(scheduler)
    else:
        client = Client()
        
    print("On initialisation, the Dask client is ", client)
    nworkers = len(client.scheduler_info()['workers'])

    sparse_graph_list = [delayed(init_sparse)(len_chunk) for i in range(nchunks)]
    psf_graph_list = [delayed(grid_and_invert_data)(s, shape) for s in sparse_graph_list]
    sum_psf_graph_rank1 = [delayed(numpy.sum)(psf_graph_list[i:i+nreduce]) for i in range(0, nchunks, nreduce)]
    sum_psf_graph = delayed(numpy.sum)(sum_psf_graph_rank1)

    future = client.compute(sum_psf_graph)
    psf = future.result()
    print(numpy.max(psf))
    
    print("At end, the Dask client is ", client)
    nworkers_final = len(client.scheduler_info()['workers'])
    assert nworkers_final == nworkers, "Lost workers: started %d, now have %d" % (nworkers, nworkers_final)
    client.shutdown()
    print("*** Successfully reached end in %.1f seconds ***" % (time.time() - start))

    exit()
