""" Benchmark processing in Dask

This mimics the overall structure and workload of our processing.

Tim Cornwell 8 November 2017
realtimcornwell@gmail.com
"""
import csv

import numpy
from dask import delayed
from distributed import Client, wait, LocalCluster

# Make some randomly located points on 2D plane
def sparse(n, margin=0.1):
    numpy.random.seed(8753193)
    return numpy.array([numpy.random.uniform(margin, 1.0 - margin, n),
                        numpy.random.uniform(margin, 1.0 - margin, n)]).reshape([n, 2])


# Put the points onto a grid and FFT
def psf(sparse_data, shape, decimate=1):
    grid = numpy.zeros(shape, dtype='complex')
    loc = numpy.round(shape * sparse_data).astype('int')
    for i in range(0, sparse_data.shape[0], decimate):
        grid[loc[i, :]] = 1.0
    return numpy.fft.fft(grid).real

def sum_list(arr):
    import copy
    result=copy.deepcopy(arr[0])
    for i, s in enumerate(arr):
        if i>0:
            result+=s
    return result

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

    # This is scaled so that for npixel=1024, it will take about 200 seconds on a single core of
    # a 2017 2.9 GHz Intel Core i7, and the time in data generation (sparse) and imaging (psf) will
    # be roughly the same.
    parser = argparse.ArgumentParser(description='Benchmark a pseudoinvert in numpy and dask')
    parser.add_argument('--scheduler', type=str, default=None,
                        help='The address of a Dask scheduler e.g. 127.0.0.1:8786. If none given one will be created')
    parser.add_argument('--decimate', type=int, default=256, help='Decimation in gridding')
    parser.add_argument('--npixel', type=int, default=1024, help='Number of pixels on a side')
    parser.add_argument('--nchunks', type=int, default=2048, help='Number of chunks of visibilities')
    parser.add_argument('--len_chunk', type=int, default=1024 * 1024,
                        help='Number of visibilities in a chunk')
    parser.add_argument('--reduce', type=int, default=16,
                        help='Gather this number of images in the first stage of the two stage summing images')
    parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--nworkers', type=int, default=None,
                        help='Number of workers if no scheduler passed in')
    parser.add_argument('--usebags', type=str, default='False', help='Use bags instead of delayed')

    args = parser.parse_args()

    results = {}
    
    # Process nchunks each of length len_chunk 2d points, making a psf of size shape
    decimate = args.decimate
    results['decimate'] = decimate
    len_chunk = args.len_chunk
    results['len_chunk'] = len_chunk
    nchunks = args.nchunks
    results['nchunks'] = nchunks
    nreduce = args.reduce
    results['reduce'] = nreduce
    npixel = args.npixel
    shape = [npixel, npixel]
    results['npixel'] = npixel
    nworkers_requested = args.nworkers
    usebags = args.usebags=='True'
    results['usebags'] = usebags

    if args.scheduler is not None:
        client = Client(args.scheduler)
    else:
        if nworkers_requested is None:
            client = Client()
        else:
            cluster = LocalCluster(n_workers=nworkers_requested, threads_per_worker=1)
            client = Client(cluster)
            
    nworkers = len(client.scheduler_info()['workers'])

    results['nworkers'] = nworkers
    results['nnodes'] = args.nnodes
    results['hostname'] = socket.gethostname()
    results['epoch'] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    print("Initial state")
    pp.pprint(results)
    
    fieldnames = ['nnodes', 'nworkers', 'time sum psf', 'npixel', 'max psf', 'len_chunk', 'nchunks', 'reduce',
                  'decimate', 'hostname', 'epoch', 'usebags']

    if usebags:
        from dask import bag
        print('Using bags to construct graph')
        sum_psf_graph = bag.from_sequence(nchunks * [len_chunk]). \
            map(sparse). \
            map(psf, shape=shape, decimate=decimate). \
            fold(numpy.add, split_every=nreduce)
    else:
        from dask import delayed
        print('Using delayed to construct graph')
        sparse_graph_list = [delayed(sparse, nout=1)(len_chunk) for i in range(nchunks)]
        psf_graph_list = [delayed(psf, nout=1)(s, shape, decimate) for s in sparse_graph_list]
        sum_psf_graph_rank1 = [delayed(sum_list, nout=1)(psf_graph_list[i:i + nreduce])
                               for i in range(0, nchunks, nreduce)]
        sum_psf_graph = delayed(sum_list)(sum_psf_graph_rank1)

    time.sleep(5)

    print("Processing graph")
    start = time.time()
    future = client.compute(sum_psf_graph)
    wait(future)
    results['time sum psf'] = time.time() - start
    
    value = future.result()
    assert value.shape[0] == shape[0] and value.shape[1] == shape[1], \
        "Shape of result %s not as requested %s" % (value.shape, shape)
    results['max psf'] = numpy.max(value)

    nworkers = len(client.scheduler_info()['workers'])
    assert nworkers == nworkers_requested, "Number of workers %d not as requested %d" % (nworkers, nworkers_requested)
    
    results['nworkers'] = nworkers
    
    filename = seqfile.findNextFile(prefix='pseudoinvert_%s_' % results['hostname'], suffix='.csv')
    print('Saving results to %s' % filename)
    write_header(filename, fieldnames)
    write_results(filename, fieldnames, results)
    client.shutdown()
    print("Final state")
    pp.pprint(results)

    exit()
