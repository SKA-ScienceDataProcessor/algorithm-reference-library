"""
Functions that aid fourier transform processing. These are built on top of the core
functions in processing_library.fourier_transforms.

The measurement equation for a sufficently narrow field of view interferometer is:

.. math::

    V(u,v,w) =\\int I(l,m) e^{-2 \\pi j (ul+vm)} dl dm


The measurement equation for a wide field of view interferometer is:

.. math::

    V(u,v,w) =\\int \\frac{I(l,m)}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm + w(\\sqrt{1-l^2-m^2}-1))} dl dm

This and related modules contain various approachs for dealing with the wide-field problem where the
extra phase term in the Fourier transform cannot be ignored.
"""

import logging
from typing import Union

import nifty_gridder as ng
import numpy
from astropy.wcs.utils import pixel_to_skycoord

from data_models.memory_data_models import Visibility, BlockVisibility, Image
from data_models.parameters import get_parameter
from ..visibility.base import copy_visibility, phaserotate_visibility

log = logging.getLogger(__name__)


def predict_ng(bvis: Union[BlockVisibility, Visibility], model: Image, gcfcf=None, **kwargs) -> \
        Union[BlockVisibility, Visibility]:
    """ Predict using convolutional degridding.
    Nifty-gridder version.

    :param bvis: BlockVisibility to be predicted
    :param model: model image
 
    :return: resulting BlockVisibility (in place works)
    """
    
    if model is None:
        return bvis

    nthreads = get_parameter(kwargs, "threads", 4)
    epsilon = get_parameter(kwargs, "epsilon", 6.0e-6)

    assert isinstance(bvis, BlockVisibility), bvis
    
    newbvis = copy_visibility(bvis, zero=True)
    
    # Extracting data from BlockVisibility
    freq = bvis.frequency  # frequency, Hz
    nants = bvis.uvw.shape[1]
    ntimes = bvis.uvw.shape[0]
    nbaselines = nants * (nants - 1) // 2
    v_nchan = bvis.vis.shape[-2]
    v_npol = bvis.vis.shape[-1]
    
    uvw = numpy.zeros([ntimes * nbaselines, 3])
    ms = numpy.zeros([ntimes * nbaselines, v_nchan, v_npol])
    
    iflat = 0
    for it in range(ntimes):
        for iant1 in range(nants):
            for iant2 in range(iant1 + 1, nants):
                uvw[iflat, :] = newbvis.data['uvw'][it, iant2, iant1, :]
                iflat += 1
    
    ms[:, :, :] = 0.0 + 0.0j  # Make all vis data equal to 0 +0j
    wgt = numpy.ones((ms.shape[0], ms.shape[2]))  # All weights equal to 1.0
    
    # Get the image properties
    m_nchan, m_npol, ny, nx = model.data.shape
    # Check if the number of frequency channels matches in bvis and a model
    assert (m_nchan == v_nchan)
    assert (m_npol == v_npol)
    
    fuvw = uvw.copy()
    # We need to flip the u and w axes. The flip in w is equivalent to the conjugation of the
    # convolution function grid_visibility to griddata
    fuvw[:, 0] *= -1.0
    fuvw[:, 2] *= -1.0
    
    # Set parameters for ng.dirty2ms()
    do_wstacking = True
    # Find out the image size/resolution
    pixsize = numpy.abs(numpy.radians(model.wcs.wcs.cdelt[0]))
    
    # Make de-gridding over a frequency range and pol fields
    for i in range(v_nchan):
        for j in range(v_npol):
            ngvis = ng.dirty2ms(fuvw.astype(numpy.float64),
                                freq[i:i + 1].astype(numpy.float64),
                                model.data[i, j, :, :].T.astype(numpy.float64),
                                wgt=wgt,
                                pixsize_x=pixsize,
                                pixsize_y=pixsize,
                                epsilon=epsilon,
                                do_wstacking=do_wstacking,
                                nthreads=nthreads,
                                verbosity=2)
            iflat = 0
            for it in range(ntimes):
                for iant1 in range(nants):
                    for iant2 in range(iant1 + 1, nants):
                        newbvis.data['vis'][it, iant2, iant1, i, j] = ngvis[iflat]
                        newbvis.data['vis'][it, iant1, iant2, i, j] = numpy.conjugate(ngvis[iflat])
                        iflat += 1
    
    # Now we can shift the visibility from the image frame to the original visibility frame
    # sbvis = shift_vis_to_image(bvis, model, tangent=True, inverse=True)
    
    return newbvis


def invert_ng(bvis: BlockVisibility, im: Image, dopsf: bool = False, normalize: bool = True, gcfcf=None, **kwargs) -> (
Image, numpy.ndarray):
    """ Invert using nifty-gridder module

    Use the image im as a template. Do PSF in a separate call.

    This is at the bottom of the layering i.e. all transforms are eventually expressed in terms
    of this function. . Any shifting needed is performed here.

    :param bvis: BlockVisibility to be inverted
    :param im: image template (not changed)
    :param normalize: Normalize by the sum of weights (True)
    :return: resulting image
    :return: sum of the weights for each frequency and polarization

    """
    
    assert isinstance(bvis, BlockVisibility), bvis
    
    nthreads = get_parameter(kwargs, "threads", 4)
    epsilon = get_parameter(kwargs, "epsilon", 6.0e-6)
    datacube = get_parameter(kwargs, "datacube", True)
    
    sbvis = copy_visibility(bvis)
    
    if dopsf:
        sbvis.data['vis'][...] = 1.0 + 0.0j
    
    # sbvis = shift_vis_to_image(sbvis, im, tangent=True, inverse=False)
    
    # Extracting data from BlockVisibility
    freq = sbvis.frequency  # frequency, Hz
    uvw_nonzero = numpy.nonzero(sbvis.uvw[:, :, :, 0])
    uvw = sbvis.uvw[uvw_nonzero]  # UVW, meters [:,3]
    ms = sbvis.vis[uvw_nonzero]  # Visibility data [:,nfreq,npol]
    #wgt = numpy.ones((ms.shape[0], ms.shape[2]))  # All weights equal to 1.0
    wgt = sbvis.imaging_weight[uvw_nonzero]
    
    # Add up XX and YY if polarized data
    if ms.shape[2] == 1:  # Scalar
        idx = [0]  # Only I
    else:  # Polar
        idx = [0, 3]  # XX and YY
    ms = numpy.sum(ms[:, :, idx], axis=2)
    wgt = numpy.sum(wgt[:, :, idx], axis=2)
    #wgt = 1 / numpy.sum(1 / wgt, axis=1)
    
    # Assign the weights to all frequencies
    #wgt = numpy.repeat(wgt[:, None], len(freq), axis=1)
    print(wgt.shape)
    do_wstacking = True
    if epsilon > 5.0e-6:
        ms = ms.astype("c8")
        wgt = wgt.astype("f4")
    
    # Find out the image size/resolution
    npixdirty = im.nwidth
    pixsize = numpy.abs(numpy.radians(im.wcs.wcs.cdelt[0]))
    
    # If non-spectral image
    if im.nchan == 1:
        datacube = False
        # Else check if the number of frequencies in the image and MS match
    else:
        assert (im.nchan == len(freq))
    
    sumwt = numpy.ones((im.nchan, im.npol))
    fuvw = uvw.copy()
    # We need to flip the u and w axes.
    fuvw[:, 0] *= -1.0
    fuvw[:, 2] *= -1.0
    if not datacube:
        dirty = ng.ms2dirty(
            fuvw, freq, ms, wgt, npixdirty, npixdirty, pixsize, pixsize, epsilon,
            do_wstacking=do_wstacking, nthreads=nthreads, verbosity=2)
        sumwt[0, 0] = numpy.sum(wgt)
        if normalize:
            dirty = dirty / sumwt[0, 0]
        im.data[0][0] = dirty.T
    else:
        for i in range(len(freq)):
            print(i, freq[i], freq[i:i + 1].shape, ms[:, i:i + 1].shape, wgt[:, i:i + 1].shape)
            dirty = ng.ms2dirty(
                fuvw, freq[i:i + 1], ms[:, i:i + 1], wgt[:, i:i + 1], npixdirty, npixdirty, pixsize, pixsize, epsilon,
                do_wstacking=do_wstacking, nthreads=nthreads, verbosity=2)
            sumwt[i, 0] = numpy.sum(wgt[:, i:i + 1])
            if normalize:
                dirty = dirty / sumwt[i, 0]
            im.data[i][0] = dirty.T
    
    return im, sumwt
