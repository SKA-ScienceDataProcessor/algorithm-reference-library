""" Benchmark processing in Dask

This mimics the overall structure and workload of our processing.

Tim Cornwell 8 November 2017
realtimcornwell@gmail.com
"""
import csv

import numpy
from dask import delayed
from distributed import Client, wait


# Make some randomly located points on 2D plane
def init_sparse(n, margin=0.1):
    numpy.random.seed(8753193)
    return numpy.array([numpy.random.uniform(margin, 1.0 - margin, n),
                        numpy.random.uniform(margin, 1.0 - margin, n)]).reshape([n, 2])


# Put the points onto a grid and FFT
def grid_and_invert_data(sparse_data, shape, decimate=1):
    grid = numpy.zeros(shape, dtype='complex')
    loc = numpy.round(shape * sparse_data).astype('int')
    for i in range(0, sparse_data.shape[0], decimate):
        grid[loc[i, :]] = 1.0
    return numpy.fft.fft(grid).real


def write_results(filename, fieldnames, results):
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writerow(results)
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
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark a pseudoinvert in numpy and dask')
    parser.add_argument('--scheduler', type=str, default=None,
                        help='The address of a Dask scheduler e.g. 127.0.0.1:8786. If none given one will be created')
    parser.add_argument('--decimate', type=int, default=1024, help='Decimation in gridding')
    parser.add_argument('--npixel', type=int, default=1024, help='Number of pixels on a side')
    parser.add_argument('--nchunks', type=int, default=256, help='Number of chunks of visibilities')
    parser.add_argument('--len_chunk', type=int, default=16384,
                        help='Number of visibilities in a chunk = len_chunk * decimate')
    parser.add_argument('--reduce', type=int, default=16,
                        help='Gather this number of images in the first stage of the two stage summing images')
    parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes')

    args = parser.parse_args()

    results = {}
    
    # Process nchunks each of length len_chunk 2d points, making a psf of size shape
    decimate = args.decimate
    results['decimate'] = decimate
    len_chunk = args.len_chunk * decimate
    results['len_chunk'] = len_chunk
    nchunks = args.nchunks
    results['nchunks'] = nchunks
    nreduce = args.reduce
    results['reduce'] = nreduce
    npixel = args.npixel
    shape = [npixel, npixel]
    results['npixel'] = npixel

    if args.scheduler is not None:
        client = Client(args.scheduler)
    else:
        client = Client()
        
    time.sleep(5)
    nworkers = len(client.scheduler_info()['workers'])

    results['nworkers'] = nworkers
    results['nnodes'] = args.nnodes
    results['hostname'] = socket.gethostname()
    results['epoch'] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    print("Initial state")
    pp.pprint(results)
    
    filename = seqfile.findNextFile(prefix='pseudoinvert_%s_' % results['hostname'], suffix='.csv')
    print('Saving results to %s' % filename)
    
    
    fieldnames = ['nnodes', 'nworkers', 'time sparse', 'time psf', 'time sum psf rank1', 'time sum psf',
                  'max psf', 'npixel', 'len_chunk', 'nchunks', 'reduce', 'decimate',
                  'hostname', 'epoch']
        
    sparse_graph_list = [delayed(init_sparse)(len_chunk) for i in range(nchunks)]
    psf_graph_list = [delayed(grid_and_invert_data)(s, shape, decimate) for s in sparse_graph_list]
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
        results['time %s' % name] = time.time() - start
        
        if name == 'sum psf':
            value = future.result()
            results['max psf'] = numpy.max(value)
        
    write_header(filename, fieldnames)
    write_results(filename, fieldnames, results)
    client.shutdown()
    print("Final state")
    pp.pprint(results)

    exit()
