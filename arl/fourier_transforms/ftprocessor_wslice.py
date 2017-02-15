# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing.
"""
import multiprocessing

import pymp

from arl.fourier_transforms.ftprocessor_base import *
from arl.image.iterators import *
from arl.image.operations import copy_image, create_empty_image_like
from arl.visibility.iterators import *
from arl.visibility.operations import create_visibility_from_rows

log = logging.getLogger(__name__)




def predict_wslice(vis, model, **kwargs):
    """ Predict using w slices.

    :param vis: Visibility to be predicted
    :param model: model image
    :param wslice: wslice in seconds. If 'auto' will find plausible value
    :param nprocessor: Number of processors to be used (1)
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_wslice: predicting using w slices")
    
    nproc = get_parameter(kwargs, "nprocessor", 1)
    if nproc == "auto":
        nproc = multiprocessing.cpu_count()
    
    nchan, npol, _, _ = model.data.shape
    
    if nproc > 1:
        
        # Extract the slices and run predict_wslice_single on each one in parallel
        rowslices = []
        wslice = get_parameter(kwargs, "wslice", 1.0)
        for rows in vis_wslice_iter(vis, wslice=wslice):
            rowslices.append(rows)
        nslices = len(rowslices)
        
        log.debug("predict_wslice: Processing %d w slices %d-way parallel" % (nslices, nproc))
        
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
                visslice = predict_wslice_single(visslice, model, **kwargs)
                with p.lock:
                    shared_vis[rows] = visslice.data['vis']
        
        vis.data['vis'][...] = shared_vis[...]

    else:
        log.debug("predict_wslice: Processing w slices serially")
        # Do each slice in turn
        wslice = get_parameter(kwargs, "wslice", 1.0)
        for rows in vis_wslice_iter(vis, wslice=wslice):
            visslice = create_visibility_from_rows(vis, rows)
            visslice = predict_wslice_single(visslice, model, **kwargs)
            vis.data['vis'][rows] += visslice.data['vis']
    
    return vis


def predict_wslice_single(vis, model, **kwargs):
    """ Predict using a single w slices.
    
    This fits a single plane and corrects the image geometry.

    :param vis: Visibility to be predicted
    :param model: model image
    :returns: resulting visibility (in place works)
    """
    log.debug("predict_wslice: predicting using w slices")
    
    vis.data['vis'] *= 0.0
    tempvis = copy_visibility(vis)

    # Calculate w beam and apply to the model. The imaginary part is not needed
    workimage = copy_image(model)
    w_beam = create_w_term_like(model, numpy.average(vis.w))
    
    # Do the real part
    workimage.data = w_beam.data.real * model.data
    vis = predict_2d(vis, workimage, **kwargs)
    
    # and now the imaginary part
    workimage.data = w_beam.data.imag * model.data
    tempvis = predict_2d(tempvis, workimage, **kwargs)
    vis.data['vis'] -= 1j * tempvis.data['vis']
    
    return vis


def invert_wslice(vis, im, dopsf=False, **kwargs):
    """ Invert using w slices (top level function)

    Use the image im as a template. Do PSF in a separate call.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param wslice: wslice in seconds. If 'auto' will find plausible value
    :param nprocessor: Number of processors to be used (1)
    :returns: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]

    """
    log.debug("invert_wslice: inverting using w slices")
    resultimage = create_empty_image_like(im)
    resultimage.data = pymp.shared.array(resultimage.data.shape)
    
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
        
        # Extract the slices and run invert_wslice_single on each one in parallel
        nslices = 0
        rowses = []
        for rows in vis_wslice_iter(vis, **kwargs):
            nslices += 1
            rowses.append(rows)
        
        log.debug("invert_wslice: Processing %d w slices %d-way parallel" % (nslices, nproc))
        with pymp.Parallel(nproc) as p:
            for index in p.range(0, nslices):
                visslice = create_visibility_from_rows(vis, rowses[index])
                workimage, sumwt = invert_wslice_single(visslice, im, dopsf, **kwargs)
                resultimage.data += workimage.data
                totalwt += sumwt
    
    else:
        # Do each slice in turn
        i = 0
        for rows in vis_wslice_iter(vis, **kwargs):
            visslice = create_visibility_from_rows(vis, rows)
            workimage, sumwt = invert_wslice_single(visslice, im, dopsf, **kwargs)
            resultimage.data += workimage.data
            totalwt += sumwt
            # export_image_to_fits(resultimage, "cumulative_image%d.fits" % (int(numpy.average(visslice.w))))
            # export_image_to_fits(workimage, "corrected_snapshot_image%d.fits" % (int(numpy.average(visslice.w))))
            i+=1
    
    return resultimage, totalwt


def invert_wslice_single(vis, im, dopsf, **kwargs):
    """Process single w slice
    
    Extracted for re-use in parallel version
    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    """
    kwargs['imaginary'] = True
    reWorkimage, sumwt, imWorkimage = invert_2d(vis, im, dopsf, **kwargs)
    # We don't normalise since that will be done after summing all images
    # export_image_to_fits(workimage, "uncorrected_snapshot_image%d.fits" % (int(numpy.average(vis.w))))

    # Calculate w beam and apply to the model. The imaginary part is not needed
    w_beam = create_w_term_like(im, numpy.average(vis.w))
    reWorkimage.data = w_beam.data.real * reWorkimage.data - w_beam.data.imag * imWorkimage.data
    
    return reWorkimage, sumwt