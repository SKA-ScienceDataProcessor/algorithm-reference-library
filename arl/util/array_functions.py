"""Useful array functions

realtimcornwell@gmail.com
"""
import numpy


def average_chunks(arr, wts, chunksize):
    """ Average the array arr with weights by chunks
    
    Array len does not have to be multiple of chunksize
    """
    if chunksize > len(arr):
        return arr, wts
    elif chunksize <= 1:
        return arr, wts
    
    places = range(0, len(arr), chunksize)
    chunks = numpy.add.reduceat(wts * arr, places)
    weights = numpy.add.reduceat(wts, places)
    
    chunks[weights > 0.0] = chunks[weights > 0.0] / weights[weights > 0.0]
    
    return chunks, weights

