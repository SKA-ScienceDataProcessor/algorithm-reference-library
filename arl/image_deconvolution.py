# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

import numpy as numpy

from crocodile.msclean import msclean
from crocodile.clean import hogbom
from arl.image_operations import create_image_from_array
from arl.data_models import *
from arl.parameters import get_parameter


def deconvolve_cube(dirty: Image, psf: Image, parameters={}):
    """ Clean using a variety of algorithms
    
    Functions that clean a dirty image using a point spread function. The algorithms available are:
    
    - Hogbom CLEAN See: Hogbom CLEAN (1974A&AS...15..417H)
    
    - MultiScale CLEAN See: Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc, 2008 vol. 2 pp. 793-801)
    
    
    :param dirty: Image dirty image
    :type Image:
    :param psf: Image Point Spread Function
    :type Image:
    :param parameters: 'algorithm': 'msclean'|'hogbom', 'gain': loop gain (float)
    :returns: componentimage, residual
    """
    algorithm = get_parameter(parameters, 'algorithm', 'msclean')
    if algorithm == 'msclean':

        window = get_parameter(parameters, 'window', None)
        gain = get_parameter(parameters, 'gain', 0.7)
        assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
        thresh = get_parameter(parameters, 'threshold', 0.0)
        assert thresh >= 0.0
        niter = get_parameter(parameters, 'niter', 100)
        assert niter > 0
        scales = get_parameter(parameters, 'scales', [0, 3, 10, 30])
        fracthresh = get_parameter(parameters, 'fracthresh', 0.01)
        assert 0.0 < fracthresh < 1.0

        comp_array = numpy.zeros(dirty.data.shape)
        residual_array = numpy.zeros(dirty.data.shape)
        for channel in range(dirty.data.shape[0]):
            for pol in range(dirty.data.shape[1]):
                if psf.data[channel, pol, :, :].max():
                    print("clean.clean: Processing pol %d, channel %d" % (pol, channel))
                    comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                       msclean(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                                window, gain, thresh, niter, scales, fracthresh)
                else:
                    print("image_deconvolution.clean: Skipping pol %d, channel %d" % (pol, channel))
    elif algorithm == 'hogbom':

        window = get_parameter(parameters, 'window', None)
        gain = get_parameter(parameters, 'gain', 0.7)
        assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
        thresh = get_parameter(parameters, 'threshold', 0.0)
        assert thresh > 0.0
        niter = get_parameter(parameters, 'niter', 100)
        assert niter > 0
        fracthresh = get_parameter(parameters, 'fracthresh', 0.01)
        assert 0.0 < fracthresh < 1.0

        comp_array = numpy.zeros(dirty.data.shape)
        residual_array = numpy.zeros(dirty.data.shape)
        for channel in range(dirty.data.shape[0]):
            for pol in range(dirty.data.shape[1]):
                if psf.data[channel, pol, :, :].max():
                    print("clean.clean: Processing pol %d, channel %d" % (pol, channel))
                    comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                        hogbom(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                               window, gain, thresh, niter)
                else:
                    print("image_deconvolution.clean: Skipping pol %d, channel %d" % (pol, channel))
    else:
        raise ValueError('image_deconvolution: Unknown algorithm %s' % algorithm)

    return create_image_from_array(comp_array, dirty.wcs), create_image_from_array(residual_array, dirty.wcs)


def restore_cube(dirty: Image, clean: Image, psf: Image, parameters={}):
    """ Restore a clean image

    :param residual: Image residual image
    :type Image:
    :param clean: Image clean model (i.e. no smoothing)
    :type Image:
    :param psf: Image Point Spread Function
    :type Image:
    :param parameters: 'algorithm': 'msclean'|'hogbom', 'gain': loop gain (float)
    :returns: restored image
    """
    print("image_deconvolution.restore_image: not yet implemented")
    return Image()


def deconvolve_mfs(dirty: Image, psf: Image, parameters={}):
    """ MFS Clean using a variety of algorithms

    Functions that clean a dirty image using a point spread function. The algorithms available are:

    - Hogbom CLEAN See: Hogbom CLEAN (1974A&AS...15..417H)

    - MultiScale CLEAN See: Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc, 2008 vol. 2 pp. 793-801)


    :param dirty: Image dirty image
    :type Image:
    :param psf: Image Point Spread Function
    :type Image:
    :param parameters: 'algorithm': 'msclean'|'hogbom', 'gain': loop gain (float)
    :returns: componentimage, residual
    """
    print("deconvolve_image.deconvolve_mfs: not yet implemented")
    return Image()


def restore_mfs(dirty: Image, clean: Image, psf: Image, parameters={}):
    """ Restore an MFS clean image

    :param residual: Image residual image
    :type Image:
    :param clean: Image clean model (i.e. no smoothing)
    :type Image:
    :param psf: Image Point Spread Function
    :type Image:
    :param parameters: 'algorithm': 'msclean'|'hogbom', 'gain': loop gain (float)
    :returns: restored image
    """
    print("deconvolve_image.restore_mfs: not yet implemented")
    return Image()
