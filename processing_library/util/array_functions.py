"""Useful array functions.

"""
import numba
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
    :return: 1D array of averaged data_models, 1d array of weights
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


@numba.jit([
            numba.types.Tuple((numba.c16[:], numba.f8[:]))(numba.c16[:], numba.f8[:], numba.i8),
            ], nopython=True)
def average_chunks_complex(arr, wts, chunksize):
    """ Average the array arr with weights by chunks

    Array len does not have to be multiple of chunksize
    The function is used for processing arr with complex data type.

    :param arr: 1D array of values
    :param wts: 1D array of weights
    :param chunksize: averaging size
    :return: 1D array of averaged data_models, 1d array of weights
    """
    if chunksize <= 1:
        return arr, wts

    mask = numpy.zeros(((len(arr) - 1) // chunksize + 1, arr.shape[0]), dtype=arr.dtype)
    maskwts = numpy.zeros_like(mask, dtype=wts.dtype)
    for enumerate_id, i in enumerate(range(0, len(arr), chunksize)):
        mask[enumerate_id, i:i + chunksize] = 1+1j
        maskwts[enumerate_id, i:i + chunksize] = 1
    tmp = wts * arr
    chunks = mask.dot(tmp)
    weights = maskwts.dot(wts)
    chunks[weights > 0.0] = chunks[weights > 0.0] / weights[weights > 0.0]

    return chunks, weights

@numba.jit([numba.types.Tuple((numba.float64[:], numba.float64[:]))(numba.float64[:], numba.float64[:], numba.int64),
            numba.types.Tuple((numba.f8[:],numba.float64[:]))(numba.f8[:], numba.float64[:], numba.int64),
            numba.types.Tuple((numba.f8[:],numba.float64[:]))(numba.f8[:], numba.float64[:], numba.int32),
            numba.types.Tuple((numba.f8[:],numba.float64[:]))(numba.f8[:], numba.float64[:], numba.int8)
            ], nopython=True)
def average_chunks(arr, wts, chunksize):
    """ Average the array arr with weights by chunks

    Array len does not have to be multiple of chunksize
    
    This version is optimised for plain numpy. It is roughly ten times faster that average_chunks_jit when used
    without numba jit. It cannot (yet) be used with numba because the add.reduceat is not support in numba
    0.31
    
    :param arr: 1D array of values
    :param wts: 1D array of weights
    :param chunksize: averaging size
    :return: 1D array of averaged data_models, 1d array of weights
    """
    if chunksize <= 1:
        return arr, wts

    # Original codes
    # places = range(0, len(arr), chunksize)
    # chunks = numpy.add.reduceat(wts * arr, places)
    # weights = numpy.add.reduceat(wts, places)
    # chunks[weights > 0.0] = chunks[weights > 0.0] / weights[weights > 0.0]

    # Codes optimized

    # mask = numpy.zeros(((len(arr)-1)//chunksize + 1, arr.shape[0]), dtype=bool)
    # for enumerate_id, i in enumerate(range(0, len(arr), chunksize)):
    #     mask[enumerate_id,i:i+chunksize]= 1
    # chunks = mask.dot(wts*arr)
    # weights = mask.dot(wts)
    # # chunks[weights > 0.0] = chunks[weights > 0.0] / weights[weights > 0.0]
    # numpy.putmask(chunks, weights>0.0, chunks/weights)

    # Numba
    mask = numpy.zeros(((len(arr)-1)//chunksize + 1, arr.shape[0]), dtype=arr.dtype)
    maskwts = numpy.zeros_like(mask,dtype=wts.dtype)
    for enumerate_id, i in enumerate(range(0, len(arr), chunksize)):
        mask[enumerate_id,i:i+chunksize]= 1
        maskwts[enumerate_id,i:i+chunksize]= 1
    tmp = wts*arr
    chunks = mask.dot(tmp)
    weights = maskwts.dot(wts)
    chunks[weights > 0.0] = chunks[weights > 0.0] / weights[weights > 0.0]

    return chunks, weights


def average_chunks2(arr, wts, chunksize):
    """ Average the two dimensional array arr with weights by chunks

    Array len does not have to be multiple of chunksize.
    
    :param arr: 2D array of values
    :param wts: 2D array of weights
    :param chunksize: 2-tuple of averaging region e.g. (2,3)
    :return: 2D array of averaged data_models, 2d array of weights
    """
    # Do each axis to determine length
    #    assert arr.shape == wts.shape, "Shapes of arrays must be the same"
    # It is possible that there is a dangling null axis on wts
    wts = wts.reshape(arr.shape)
    #
    # # For numba
    if arr.dtype=='c16':
        l0 = len(average_chunks_complex(arr[:, 0].flatten(), wts[:, 0].flatten(), chunksize[0])[0])
        l1 = len(average_chunks_complex(arr[0, :].flatten(), wts[0, :].flatten(), chunksize[1])[0])

        tempchunks = numpy.zeros([arr.shape[0], l1], dtype=arr.dtype)
        tempwt = numpy.zeros([arr.shape[0], l1])

        tempchunks *= tempwt
        for i in range(arr.shape[0]):
            result = average_chunks_complex(arr[i, :], wts[i, :], chunksize[1])
            tempchunks[i, :], tempwt[i, :] = result[0].flatten(), result[1].flatten()

        chunks = numpy.zeros([l0, l1], dtype=arr.dtype)
        weights = numpy.zeros([l0, l1])

        for i in range(l1):
            result = average_chunks_complex(tempchunks[:, i], tempwt[:, i], chunksize[0])
            chunks[:, i], weights[:, i] = result[0].flatten(), result[1].flatten()

    else:
        l0 = len(average_chunks(arr[:, 0].flatten(), wts[:, 0].flatten(), chunksize[0])[0])
        l1 = len(average_chunks(arr[0, :].flatten(), wts[0, :].flatten(), chunksize[1])[0])

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


    # Original
    # l0 = len(average_chunks(arr[:, 0].flatten(), wts[:, 0].flatten(), chunksize[0])[0])
    # l1 = len(average_chunks(arr[0, :].flatten(), wts[0, :].flatten(), chunksize[1])[0])
    #
    # tempchunks = numpy.zeros([arr.shape[0], l1], dtype=arr.dtype)
    # tempwt = numpy.zeros([arr.shape[0], l1])
    #
    # tempchunks *= tempwt
    # for i in range(arr.shape[0]):
    #     result = average_chunks(arr[i, :], wts[i, :], chunksize[1])
    #     tempchunks[i, :], tempwt[i, :] = result[0].flatten(), result[1].flatten()
    #
    # chunks = numpy.zeros([l0, l1], dtype=arr.dtype)
    # weights = numpy.zeros([l0, l1])
    #
    # for i in range(l1):
    #     result = average_chunks(tempchunks[:, i], tempwt[:, i], chunksize[0])
    #     chunks[:, i], weights[:, i] = result[0].flatten(), result[1].flatten()

    return chunks, weights


def tukey_filter(x, r):
    """ Calculate the Tukey (tapered cosine) filter
    
    See e.g. https://uk.mathworks.com/help/signal/ref/tukeywin.html

    :param x: x coordinate (float)
    :param r: transition point of filter (float)
    :returns: Value of filter for x
    """
    if 0.0 <= x < r / 2.0:
        return 0.5 * (1.0 + numpy.cos(2.0 * numpy.pi * (x - r / 2.0) / r))
    elif 1 - r / 2.0 <= x <= 1.0:
        return 0.5 * (1.0 + numpy.cos(2.0 * numpy.pi * (x - 1 + r / 2.0) / r))
    else:
        return 1.0


def insert_function_sinc(x):
    s = numpy.zeros_like(x)
    s[x != 0.0] = numpy.sin(numpy.pi * x[x != 0.0]) / (numpy.pi * x[x != 0.0])
    return s


def insert_function_L(x, a=5):
    L = insert_function_sinc(x) * insert_function_sinc(x / a)
    return L


def insert_function_pswf(x, a=5):
    from processing_library.fourier_transforms.convolutional_gridding import grdsf
    return grdsf(abs(x) / a)[1]


def insert_array(im, x, y, flux, bandwidth=1.0, support=7, insert_function=insert_function_L):
    """ Insert point into image using specified function
    
    :param im: Image
    :param x: x in float pixels
    :param y: y in float pixels
    :param flux: Flux[nchan, npol]
    :param bandwidth: Support of data in uv plane
    :param support: Support of function in image space
    :param insert_function: insert_function_L or insert_function_Sinc or insert_function_pswf
    :return:
    """
    nchan, npol, ny, nx = im.shape
    intx = int(numpy.round(x))
    inty = int(numpy.round(y))
    fracx = x - intx
    fracy = y - inty
    gridx = numpy.arange(-support, support)
    gridy = numpy.arange(-support, support)
    
    insert = numpy.outer(insert_function(bandwidth * (gridy - fracy)),
                         insert_function(bandwidth * (gridx - fracx)))
    
    insertsum = numpy.sum(insert)
    assert insertsum > 0, "Sum of interpolation coefficients %g" % insertsum
    insert = insert / insertsum
    
    for chan in range(nchan):
        for pol in range(npol):
            im[chan, pol, inty - support:inty + support, intx - support:intx + support] += flux[chan, pol] * insert
    
    return im
