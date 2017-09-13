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


# Put the points onto a grid and FFT, skip to save time
def grid_data(sparse_data, shape, skip=100):
    grid = numpy.zeros(shape, dtype='complex')
    loc = numpy.round(shape * sparse_data).astype('int')
    for i in range(0, sparse_data.shape[0], skip):
        grid[loc[i,:]] = 1.0
    return numpy.fft.fft(grid).real

# Accumulate all psfs into one psf
def accumulate(psf_list):
    lpsf = 0.0 * psf_list[0]
    for p in psf_list:
        lpsf += p
    return lpsf


if __name__ == '__main__':
    import sys
    import time
    start=time.time()
    
    # Process nchunks each of length len_chunk 2d points, making a psf of size shape
    len_chunk = int(1e6)
    nchunks = 16
    shape=[512, 512]
    skip = 100
    
    # We pass in the scheduler from the invoking script
    if len(sys.argv) > 1:
        scheduler = sys.argv[1]
        client = Client(scheduler)
    else:
        client = Client()
        
    print("On initialisation", client)

    sparse_graph = [delayed(init_sparse)(len_chunk) for i in range(nchunks)]
    sparse_graph = client.compute(sparse_graph, sync=True)
    print("After first sparse_graph", client)
    
    xfr_graph = [delayed(grid_data)(s, shape=shape, skip=skip) for s in sparse_graph]
    xfr = client.compute(xfr_graph, sync=True)
    print("After xfr", client)

    tsleep = 120.0
    print("Sleeping now for %.1f seconds" % tsleep)
    time.sleep(tsleep)
    print("After sleep", client)

    sparse_graph = [delayed(init_sparse)(len_chunk) for i in range(nchunks)]
    # sparse_graph = client.compute(sparse_graph, sync=True)
    xfr_graph = [delayed(grid_data)(s, shape=shape, skip=skip) for s in sparse_graph]
    psf_graph = delayed(accumulate)(xfr_graph)
    psf = client.compute(psf_graph, sync=True)
    
    print("*** Successfully reached end in %.1f seconds ***" % (time.time() - start))
    print(numpy.max(psf))
    print("After psf", client)

    client.shutdown()
    exit()
