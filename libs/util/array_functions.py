"""Useful array functions.

"""
# import numba
import numpy


# @numba.jit([numba.types.Tuple((numba.float64[:], numba.float64[:]))
#                 (numba.float64[:], numba.float64[:], numba.int64)], nopython=True)
def average_chunks_jit(arr, wts, chunksize):
    """ Average the array arr with weights by chunks

    Array len does not have to be multiple of chunksize
    
    This is a version written for numba. When used with numba.jit, it's about 25 - 30% faster than the
    numpy version without jit.
    
    :param arr: 1D array of values
    :param wts: 1D array of weights
    :param chunksize: averaging size
    :return: 1D array of averaged data, 1d array of weights
    """
    if chunksize <= 1:
        return arr, wts
    nchunks = len(arr) // chunksize
    extra = len(arr) % chunksize
    if extra > 0:
        fullsize = nchunks + 1
    else:
        fullsize = nchunks
    
    chunks = numpy.empty(fullsize, dtype=arr.dtype)
    weights = numpy.empty(fullsize, dtype=wts.dtype)
    
    for place in range(nchunks):
        chunks[place] = numpy.sum(
            wts[place * chunksize:(place + 1) * chunksize] * arr[place * chunksize:(place + 1) * chunksize])
        weights[place] = numpy.sum(wts[place * chunksize:(place + 1) * chunksize])
    
    if extra > 0:
        chunks[-1] = numpy.sum(wts[(len(arr) - extra):len(arr)] * arr[(len(arr) - extra):len(arr)])
        weights[-1] = numpy.sum(wts[(len(arr) - extra):len(arr)])
    
    chunks[weights > 0.0] = chunks[weights > 0.0] / weights[weights > 0.0]
    
    return chunks, weights


def average_chunks(arr, wts, chunksize):
    """ Average the array arr with weights by chunks

    Array len does not have to be multiple of chunksize
    
    This version is optimised for plain numpy. It is roughly ten times faster that average_chunks_jit when used
    without numba jit. It cannot (yet) be used with numba because the add.reduceat is not support in numba
    0.31
    
    :param arr: 1D array of values
    :param wts: 1D array of weights
    :param chunksize: averaging size
    :return: 1D array of averaged data, 1d array of weights
    """
    if chunksize <= 1:
        return arr, wts
    
    places = range(0, len(arr), chunksize)
    chunks = numpy.add.reduceat(wts * arr, places)
    weights = numpy.add.reduceat(wts, places)
    
    chunks[weights > 0.0] = chunks[weights > 0.0] / weights[weights > 0.0]
    
    return chunks, weights


def average_chunks2(arr, wts, chunksize):
    """ Average the two dimensional array arr with weights by chunks

    Array len does not have to be multiple of chunksize.
    
    :param arr: 2D array of values
    :param wts: 2D array of weights
    :param chunksize: 2-tuple of averaging region e.g. (2,3)
    :return: 2D array of averaged data, 2d array of weights
    """
    # Do each axis to determine length
    #    assert arr.shape == wts.shape, "Shapes of arrays must be the same"
    # It is possible that there is a dangling null axis on wts
    wts = wts.reshape(arr.shape)
    
    l0 = len(average_chunks(arr[:, 0], wts[:, 0], chunksize[0])[0])
    l1 = len(average_chunks(arr[0, :], wts[0, :], chunksize[1])[0])
    
    tempchunks = numpy.zeros([arr.shape[0], l1], dtype=arr.dtype)
    tempwt = numpy.zeros([arr.shape[0], l1])
    
    tempchunks *= tempwt
    for i in range(arr.shape[0]):
        result = average_chunks(arr[i, :], wts[i, :], chunksize[1])
        tempchunks[i, :], tempwt[i, :] = result[0].flatten(), result[1].flatten()
    
    chunks = numpy.zeros([l0, l1], dtype=arr.dtype)
    weights = numpy.zeros([l0, l1])
    
    for i in range(l1):
        result = average_chunks(tempchunks[:, i], tempwt[:, i], chunksize[0])
        chunks[:, i], weights[:, i] = result[0].flatten(), result[1].flatten()
    
    return chunks, weights
