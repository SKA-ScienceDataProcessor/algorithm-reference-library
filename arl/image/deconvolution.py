# Tim Cornwell <realtimcornwell@gmail.com>
""" Image Deconvolution functions

"""

import numpy
import logging

from astropy.convolution import Gaussian2DKernel, convolve
from photutils import fit_2dgaussian

from arl.data.data_models import Image
from arl.data.parameters import get_parameter
from arl.image.operations import create_image_from_array, copy_image, calculate_image_frequency_moments, smooth_image

from arl.image.hogbom import hogbom
from arl.image.msclean import msclean

log = logging.getLogger(__name__)

def deconvolve_cube(dirty: Image, psf: Image, **kwargs):
    """ Clean using a variety of algorithms
    
    Functions that clean a dirty image using a point spread function. The algorithms available are:
    
    - Hogbom CLEAN See: Hogbom CLEAN (1974A&AS...15..417H)
    
    - MultiScale CLEAN See: Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc, 2008 vol. 2 pp. 793-801)
    
    
    :param dirty: Image dirty image
    :param psf: Image Point Spread Function
    :param window: Window image (Bool) - clean where True
    :param algorithm: Cleaning algorithm: 'msclean'|'hogbom'
    :param gain: loop gain (float) 0.7
    :param threshold: Clean threshold (0.0)
    :param fracthres: Fractional threshold (0.01)
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
    :returns: componentimage, residual
    """
    
    window = get_parameter(kwargs, 'window', None)
    if window == 'quarter':
        qx = dirty.shape[3] // 4
        qy = dirty.shape[2] // 4
        window = numpy.zeros_like(dirty.data)
        window[..., (qy + 1):3 * qy, (qx + 1):3 * qx] = 1.0
        log.info('deconvolve_cube: Cleaning inner quarter of each sky plane')
    else:
        window = None
    
    psf_support = get_parameter(kwargs, 'psf_support', None)
    if isinstance(psf_support, int):
        if (psf_support < psf.shape[2] // 2) and ((psf_support < psf.shape[3] // 2)):
            centre = [psf.shape[2] // 2, psf.shape[3] // 2]
            psf.data = psf.data[..., (centre[0] - psf_support):(centre[0] + psf_support),
                       (centre[1] - psf_support):(centre[1] + psf_support)]
            log.info('deconvolve_cube: PSF support = +/- %d pixels' % (psf_support))
    
    algorithm = get_parameter(kwargs, 'algorithm', 'msclean')
    if algorithm == 'msclean':
        
        gain = get_parameter(kwargs, 'gain', 0.7)
        assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
        thresh = get_parameter(kwargs, 'threshold', 0.0)
        assert thresh >= 0.0
        niter = get_parameter(kwargs, 'niter', 100)
        assert niter > 0
        scales = get_parameter(kwargs, 'scales', [0, 3, 10, 30])
        fracthresh = get_parameter(kwargs, 'fractional_threshold', 0.01)
        assert 0.0 < fracthresh < 1.0
        
        comp_array = numpy.zeros(dirty.data.shape)
        residual_array = numpy.zeros(dirty.data.shape)
        for channel in range(dirty.data.shape[0]):
            for pol in range(dirty.data.shape[1]):
                if psf.data[channel, pol, :, :].max():
                    log.info("deconvolve_cube: Processing pol %d, channel %d" % (pol, channel))
                    if window is None:
                        comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                            msclean(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                                    None, gain, thresh, niter, scales, fracthresh)
                    else:
                        comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                            msclean(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                                    window[channel, pol, :, :], gain, thresh, niter, scales, fracthresh)
                else:
                    log.info("deconvolve_cube: Skipping pol %d, channel %d" % (pol, channel))
    elif algorithm == 'hogbom':
        
        gain = get_parameter(kwargs, 'gain', 0.7)
        assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
        thresh = get_parameter(kwargs, 'threshold', 0.0)
        assert thresh >= 0.0
        niter = get_parameter(kwargs, 'niter', 100)
        assert niter > 0
        fracthresh = get_parameter(kwargs, 'fractional_threshold', 0.1)
        assert 0.0 <= fracthresh < 1.0
        
        comp_array = numpy.zeros(dirty.data.shape)
        residual_array = numpy.zeros(dirty.data.shape)
        for channel in range(dirty.data.shape[0]):
            for pol in range(dirty.data.shape[1]):
                if psf.data[channel, pol, :, :].max():
                    log.info("deconvolve_cube: Processing pol %d, channel %d" % (pol, channel))
                    if window is None:
                        comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                            hogbom(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                                   None, gain, thresh, niter, fracthresh)
                    else:
                        comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                            hogbom(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                                   window[channel, pol, :, :], gain, thresh, niter, fracthresh)
                else:
                    log.info("deconvolve_cube: Skipping pol %d, channel %d" % (pol, channel))
    else:
        raise ValueError('deconvolve_cube: Unknown algorithm %s' % algorithm)
    
    return create_image_from_array(comp_array, dirty.wcs), create_image_from_array(residual_array, dirty.wcs)


def deconvolve_mfs(dirty: Image, psf: Image, **kwargs):
    """ MFS Clean using a variety of algorithms

    Functions that clean a dirty image using a point spread function. The algorithms available are:

    - Hogbom CLEAN See: Hogbom CLEAN (1974A&AS...15..417H)

    - MultiScale CLEAN See: Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc, 2008 vol. 2 pp. 793-801)


    :param dirty: Image dirty image
    :param psf: Image Point Spread Function
    :param params: 'algorithm': 'msclean'|'hogbom', 'gain': loop gain (float)
    :returns: componentimage, residual
    """
    nmoments = get_parameter(kwargs, "nmoments", 3)
    dirty_taylor = calculate_image_frequency_moments(dirty, nmoments=nmoments)
    psf_taylor = calculate_image_frequency_moments(psf, nmoments=nmoments)

    raise ValueError("deconvolve_mfs: not yet implemented")


def restore_mfs(dirty: Image, clean: Image, psf: Image, **kwargs):
    """ Restore an MFS clean image

    :param dirty:
    :param clean: Image clean model (i.e. no smoothing)
    :param psf: Image Point Spread Function
    :param params: 'algorithm': 'msclean'|'hogbom', 'gain': loop gain (float)
    :returns: restored image
    """
    
    raise ValueError("restore_mfs: not yet implemented")


def restore_cube(model, psf, residual=None):
    """ Restore the model image to the residuals

    :params psf: Input PSF
    :returns: major axis, minor axis, position angle (in radians)

    """
    restored = copy_image(model)
    
    npixel = psf.data.shape[3]
    sl = slice(npixel // 2 - 7, npixel // 2 + 8)
    
    # isotropic at the moment!
    try:
        fit = fit_2dgaussian(psf.data[0, 0, sl, sl])
        if fit.x_stddev <= 0.0 or fit.y_stddev <= 0.0:
            log.debug('restore_cube: error in fitting to psf, using 1 pixel stddev')
            size = 1.0
        else:
            size = max(fit.x_stddev, fit.y_stddev)
            log.debug('restore_cube: psfwidth = %s' % (size))
    except:
        log.debug('restore_cube: warning in fit to psf, using 1 pixel stddev')
        size = 1.0
    
    # By convention, we normalise the peak not the integral so this is the volume of the Gaussian
    norm = 2.0 * numpy.pi * size ** 2
    gk = Gaussian2DKernel(size)
    for chan in range(model.shape[0]):
        for pol in range(model.shape[1]):
            restored.data[chan, pol, :, :] = norm * convolve(model.data[chan, pol, :, :], gk, normalize_kernel=False)
    if residual is not None:
        restored.data += residual.data
    return restored
