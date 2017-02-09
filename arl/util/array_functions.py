"""Useful array functions

realtimcornwell@gmail.com
"""
import numpy


def average_chunks(arr, wts, chunksize):
    """ Average the array arr with weights by chunks

    Array len does not have to be multiple of chunksize
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

    Array len does not have to be multiple of chunksize
    """
    # Do each axis to determine length
#    assert arr.shape == wts.shape, "Shapes of arrays must be the same"
    # It is possible that there is a dangling null axis on wts
    wts = wts.reshape(arr.shape)
    
    l0 = len(average_chunks(arr[:,0], wts[:,0], chunksize[0])[0])
    l1 = len(average_chunks(arr[0,:], wts[0,:], chunksize[1])[0])

    tempchunks = numpy.zeros([arr.shape[0], l1], dtype=arr.dtype)
    tempwt = numpy.zeros([arr.shape[0], l1])

    tempchunks *= tempwt
    for i in range(arr.shape[0]):
        result = average_chunks(arr[i,:], wts[i,:], chunksize[1])
        tempchunks[i,:], tempwt[i,:] = result[0].flatten(), result[1].flatten()

    chunks = numpy.zeros([l0, l1], dtype=arr.dtype)
    weights = numpy.zeros([l0, l1])

    for i in range(l1):
        result = average_chunks(tempchunks[:, i], tempwt[:, i], chunksize[0])
        chunks[:,i], weights[:,i] = result[0].flatten(), result[1].flatten()

    return chunks, weights

