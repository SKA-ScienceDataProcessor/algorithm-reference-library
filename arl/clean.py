# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

import numpy as numpy

from crocodile.msclean import msclean
from crocodile.clean import hogbom
from arl.image import Image, image_from_array


def clean(dirty: Image, psf: Image, **kwargs):
    """ Clean using a variety of algorithms
    
    Functions that clean a dirty image using a point spread function. The algorithms available are:
    
    - Hogbom CLEAN See: Hogbom CLEAN (1974A&AS...15..417H)
    
    - MultiScale CLEAN See: Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc, 2008 vol. 2 pp. 793-801)
    
    :param dirty: Image dirty image
    :type Image:
    :param psf: Image Point Spread Function
    :type Image:
    :param kwargs: 'algorithm': 'msclean'|'hogbom', 'gain': loop gain (float)
    :returns: componentimage, residual
    """
    algorithm = kwargs.get('algorithm', 'msclean')
    if algorithm == 'msclean':

        window = kwargs.get('window', None)
        gain = kwargs.get('gain', 0.7)
        assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
        thresh = kwargs.get('threshold', 0.0)
        assert thresh > 0.0
        niter = kwargs.get('niter', 100)
        assert niter > 0
        scales = kwargs.get('scales', [0, 3, 10, 30])
        fracthresh = kwargs.get('fracthresh', 0.0)
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
                    print("clean.clean: Skipping pol %d, channel %d" % (pol, channel))
    elif algorithm == 'hogbom':

        window = kwargs.get('window', None)
        gain = kwargs.get('gain', 0.7)
        assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
        thresh = kwargs.get('threshold', 0.0)
        assert thresh > 0.0
        niter = kwargs.get('niter', 100)
        assert niter > 0
        fracthresh = kwargs.get('fracthresh', 0.01)
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
                    print("clean.clean: Skipping pol %d, channel %d" % (pol, channel))
    else:
        raise ValueError('Unknown algorithm %s' % algorithm)

    return image_from_array(comp_array, dirty.wcs), image_from_array(residual_array, dirty.wcs)
