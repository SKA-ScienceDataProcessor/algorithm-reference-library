""" Benchmark processing in Dask

This mimics the overall structure and workload of our processing.

Tim Cornwell 8 November 2017
realtimcornwell@gmail.com
"""
import csv

import numpy
from dask import delayed
from distributed import Client, wait, Scheduler


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
        grid[loc[i, :]] = 1.0
    return numpy.fft.fft(grid).real


def write_results(filename, fieldnames, results):
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writerow(results)
        print(results)
        csvfile.close()


def write_header(filename, fieldnames):
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        csvfile.close()


if __name__ == '__main__':
    import time
    import socket
    import seqfile

    import sys

    results = {}
    if len(sys.argv) > 1:
        client = Client(sys.argv[1])
    else:
        client = Client()
        
    time.sleep(5)
    nworkers = len(client.scheduler_info()['workers'])
    print("Using %d workers" % (nworkers))

    results['nworkers'] = nworkers
    results['hostname'] = socket.gethostname()
    results['epoch'] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    start = time.time()
    filename = seqfile.findNextFile(prefix='pseudoinvert_%s_' % results['hostname'], suffix='.csv')
    print('Saving results to %s' % filename)
    
    # Process nchunks each of length len_chunk 2d points, making a psf of size shape
    len_chunk = 16384
    results['len_chunk'] = len_chunk
    nchunks = 256
    results['nchunks'] = nchunks
    nreduce = 16
    results['reduce'] = nreduce
    npixel = 1024
    shape = [npixel, npixel]
    results['npixel'] = npixel
    
    fieldnames = ['nworkers', 'time sparse', 'time psf', 'time sum psf rank1', 'time sum psf',
                  'max psf', 'npixel', 'len_chunk', 'nchunks', 'reduce', 'hostname', 'epoch']
        
    sparse_graph_list = [delayed(init_sparse)(len_chunk) for i in range(nchunks)]
    psf_graph_list = [delayed(grid_and_invert_data)(s, shape) for s in sparse_graph_list]
    sum_psf_graph_rank1 = [delayed(numpy.sum)(psf_graph_list[i:i + nreduce]) for i in range(0, nchunks, nreduce)]
    sum_psf_graph = delayed(numpy.sum)(sum_psf_graph_rank1)
    sum_psf_graph.visualise()

    names = ['sparse', 'psf', 'sum psf rank1', 'sum psf']
    graphs = [sparse_graph_list, psf_graph_list, sum_psf_graph_rank1, sum_psf_graph]
    for graph, name in zip(graphs, names):
        print("Processing graph %s" % name)
        start = time.time()
        future = client.compute(graph)
        wait(future)
        if isinstance(future, list):
            value = [f.result() for f in future]
        else:
            value = future.result()
        results['time %s' % name] = time.time() - start
        
        if name == 'sum psf':
            results['max psf'] = numpy.max(value)
        
    print(fieldnames)
    print(results.keys())
    write_header(filename, fieldnames)
    write_results(filename, fieldnames, results)
    client.shutdown()
    
    exit()
