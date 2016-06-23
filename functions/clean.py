# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

import numpy as numpy

from crocodile.msclean import msclean
from functions.image import Image, image_from_array


def clean(dirty: Image, psf: Image, **kwargs):
    """

    :param dirty:
    :param psf:
    :param kwargs:
    :return:
    """
    algorithm = kwargs.get('algorithm', 'msclean')
    if algorithm == 'msclean':

        window = kwargs.get('window', None)
        gain = kwargs.get('gain', 0.7)
        thresh = kwargs.get('threshold', 0.0)
        niter = kwargs.get('niter', 100)
        scales = kwargs.get('scales', [0, 3, 10, 30])
        fracthresh = kwargs.get('fracthresh', 0.0)
        comp_array = numpy.zeros(dirty.data.shape)
        residual_array = numpy.zeros(dirty.data.shape)
        for channel in range(dirty.data.shape[0]):
            for pol in range(dirty.data.shape[1]):
                print("clean.clean: Processing pol %d, channel %d" % (pol, channel))
                comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                    msclean(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                            window, gain, thresh, niter, scales, fracthresh)
    else:
        raise ValueError('Unknown algorithm %s' % algorithm)

    return image_from_array(comp_array, dirty.wcs), image_from_array(residual_array, dirty.wcs)
