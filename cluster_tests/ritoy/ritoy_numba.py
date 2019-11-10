# """ Radio interferometry toy
#
# This mimics the overall structure and workload of our processing.
#
# Tim Cornwell 9 Sept 2017
# realtimcornwell@gmail.com
# Adding Numba Testing, ^W_F^
# """
import numpy
from dask import delayed
from distributed import Client
import numba

# Make some randomly located points on 2D plane

@numba.jit('f8[:,:](i8,f8)',nopython=True)
def init_sparse_pre(n, margin):
    numpy.random.seed(8753193)
    # a = numpy.array([numpy.random.uniform(margin, 1.0 - margin, n),
    #                     numpy.random.uniform(margin, 1.0 - margin, n)]).reshape([n, 2])
    a = numpy.random.uniform(margin, 1.0 - margin, 2*n)
    a = a.astype(numpy.float64)
    a = a.reshape(-1,2)
    return a

def init_sparse(n, margin=0.1):
    return init_sparse_pre(n, margin)

# Put the points onto a grid and FFT
@numba.jit('c16[:,:](f8[:,:])',nopython=True)
def grid_and_invert_data_pre(sparse_data):
    shape=[1024, 1024]
    grid = numpy.zeros((1024,1024), dtype=numpy.complex128)
    loc = numpy.array([1024.,1024.]) * sparse_data
    out = numpy.empty_like(loc)
    loc = numpy.round_(loc,0,out).astype(numpy.int64)
    for i in range(0, sparse_data.shape[0]):
        grid[loc[i,:]] = 1.0
    return(grid)

def grid_and_invert_data(sparse_data, shape):
    grid = grid_and_invert_data_pre(sparse_data)
    return numpy.fft.fft(grid).real

if __name__ == '__main__':
    import sys
    import time

    start=time.time()

    # Process nchunks each of length len_chunk 2d points, making a psf of size shape
    len_chunk = 16384*8
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

    sparse_graph_list = [delayed(init_sparse)(len_chunk) for i in range(nchunks)]
    psf_graph_list = [delayed(grid_and_invert_data)(s, shape) for s in sparse_graph_list]
    sum_psf_graph_rank1 = [delayed(numpy.sum)(psf_graph_list[i:i+nreduce]) for i in range(0, nchunks, nreduce)]
    sum_psf_graph = delayed(numpy.sum)(sum_psf_graph_rank1)

    future = client.compute(sum_psf_graph)
    psf = future.result()
    print(numpy.max(psf))

    client.close()
    print("*** Successfully reached end in %.1f seconds ***" % (time.time() - start))

    exit(0)
