# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that distributes predict and invert using either just loops or parallel execution
"""

import multiprocessing
import pymp


from arl.fourier_transforms.ftprocessor_params import *
from arl.fourier_transforms.ftprocessor_base import *
from arl.image.iterators import *
from arl.image.operations import create_empty_image_like
from arl.visibility.iterators import vis_slice_iter
from arl.visibility.operations import create_visibility_from_rows
from arl.data.parameters import get_parameter


log = logging.getLogger(__name__)

def invert_with_iterator(vis, im, dopsf=False, vis_iter=vis_slice_iter, invert=invert_2d, **kwargs):
    """ Invert using a specified iterator and invert
    
    This knows about the structure of invert in different execution frameworks but not
    anything about the actual processing. This version support pymp and serial processing

    :param vis:
    :param im:
    :param dopsf:
    :param kwargs:
    :return:
    """
    resultimage = create_empty_image_like(im)
    
    nproc = get_parameter(kwargs, "nprocessor", 1)
    if nproc == "auto":
        nproc = multiprocessing.cpu_count()
    inchan, inpol, _, _ = im.data.shape
    totalwt = numpy.zeros([inchan, inpol], dtype='float')
    if nproc > 1:
        # We need to tell pymp that some arrays are shared
        resultimage.data = pymp.shared.array(resultimage.data.shape)
        resultimage.data *= 0.0
        totalwt = pymp.shared.array([inchan, inpol])
        
        # Extract the slices and run  on each one in parallel
        nslices = 0
        rowses = []
        for rows in vis_iter(vis, **kwargs):
            nslices += 1
            rowses.append(rows)
        
        log.debug("invert_iteratoe: Processing %d chunks %d-way parallel" % (nslices, nproc))
        with pymp.Parallel(nproc) as p:
            for index in p.range(0, nslices):
                visslice = create_visibility_from_rows(vis, rowses[index])
                workimage, sumwt = invert(visslice, im, dopsf, **kwargs)
                resultimage.data += workimage.data
                totalwt += sumwt
    
    else:
        # Do each slice in turn
        i = 0
        for rows in vis_iter(vis, **kwargs):
            visslice = create_visibility_from_rows(vis, rows)
            workimage, sumwt = invert(visslice, im, dopsf, **kwargs)
            resultimage.data += workimage.data
            totalwt += sumwt
            i += 1
    return resultimage, totalwt


def predict_with_iterator(vis, model, vis_iter=vis_slice_iter, predict=predict_2d, **kwargs):
    """Iterate through prediction in chunks
    
    This knows about the structure of predict in different execution frameworks but not
    anything about the actual processing. This version support pymp and serial processing
    
    """
    nproc = get_parameter(kwargs, "nprocessor", 1)
    if nproc == "auto":
        nproc = multiprocessing.cpu_count()
    nchan, npol, _, _ = model.data.shape
    if nproc > 1:
        
        # Extract the slices and run predict on each one in parallel
        rowslices = []
        for rows in vis_iter(vis, **kwargs):
            rowslices.append(rows)
        nslices = len(rowslices)
        
        log.debug("predict_with_iterator: Processing %d chunks %d-way parallel" % (nslices, nproc))
        
        # The visibility column needs to be shared across all processes
        # We have to work around lack of complex data in pymp. For the following trick, see
        # http://stackoverflow.com/questions/2598734/numpy-creating-a-complex-array-from-2-real-ones
        
        shape = vis.data['vis'].shape
        shape = [shape[0], shape[1], shape[2], 2]
        log.debug('Creating shared array of float type and shape %s for visibility' % (str(shape)))
        shared_vis = pymp.shared.array(shape).view(dtype='complex128')[..., 0]
        
        with pymp.Parallel(nproc) as p:
            for slice in p.range(0, nslices):
                rows = rowslices[slice]
                visslice = create_visibility_from_rows(vis, rows)
                visslice = predict(visslice, model, **kwargs)
                with p.lock:
                    shared_vis[rows] = visslice.data['vis']
        
        vis.data['vis'][...] = shared_vis[...]
    
    else:
        log.debug("predict_with_iterator: Processing chunks serially")
        # Do each chunk in turn
        for rows in vis_iter(vis, **kwargs):
            visslice = create_visibility_from_rows(vis, rows)
            visslice = predict(visslice, model, **kwargs)
            vis.data['vis'][rows] += visslice.data['vis']
    return vis

