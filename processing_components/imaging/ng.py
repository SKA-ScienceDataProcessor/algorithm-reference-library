"""
Functions that implement prediction of and imaging from visibilities using the nifty gridder.

https://gitlab.mpcdf.mpg.de/ift/nifty_gridder

This performs all necessary w term corrections, to high precision.

"""

__all__ = ['predict_ng', 'invert_ng']

import logging
from typing import Union

import numpy

from data_models.memory_data_models import Visibility, BlockVisibility, Image
from data_models.parameters import get_parameter
from data_models.polarisation import convert_pol_frame
from processing_components.image.operations import copy_image
from processing_components.imaging.base import shift_vis_to_image, normalize_sumwt
from processing_components.visibility.base import copy_visibility

log = logging.getLogger(__name__)

try:
    import nifty_gridder as ng
    
    def predict_ng(bvis: Union[BlockVisibility, Visibility], model: Image, **kwargs) -> \
            Union[BlockVisibility, Visibility]:
        """ Predict using convolutional degridding.
        
        Nifty-gridder version. https://gitlab.mpcdf.mpg.de/ift/nifty_gridder
    
        :param bvis: BlockVisibility to be predicted
        :param model: model image
        :return: resulting BlockVisibility (in place works)
        """

        assert isinstance(bvis, BlockVisibility), bvis
        

        if model is None:
            return bvis
        
        nthreads = get_parameter(kwargs, "threads", 4)
        epsilon = get_parameter(kwargs, "epsilon", 1e-12)
        do_wstacking = get_parameter(kwargs, "do_wstacking", True)
        verbosity = get_parameter(kwargs, "verbosity", 2)
        
        newbvis = copy_visibility(bvis, zero=True)
        
        # Extracting data from BlockVisibility
        freq = bvis.frequency  # frequency, Hz
        nrows, nants, _, vnchan, vnpol = bvis.vis.shape
        
        uvw = newbvis.data['uvw'].reshape([nrows * nants * nants, 3])
        vis = newbvis.data['vis'].reshape([nrows * nants * nants, vnchan, vnpol])
        
        vis[...] = 0.0 + 0.0j  # Make all vis data equal to 0 +0j
        
        # Get the image properties
        m_nchan, m_npol, ny, nx = model.data.shape
        # Check if the number of frequency channels matches in bvis and a model
        #        assert (m_nchan == v_nchan)
        assert (m_npol == vnpol)
        
        fuvw = uvw.copy()
        # We need to flip the u and w axes. The flip in w is equivalent to the conjugation of the
        # convolution function grid_visibility to griddata
        fuvw[:, 0] *= -1.0
        fuvw[:, 2] *= -1.0
        
        # Find out the image size/resolution
        pixsize = numpy.abs(numpy.radians(model.wcs.wcs.cdelt[0]))
        
        # Make de-gridding over a frequency range and pol fields
        vis_to_im = numpy.round(model.wcs.sub([4]).wcs_world2pix(freq, 0)[0]).astype('int')
        for vchan in range(vnchan):
            imchan = vis_to_im[vchan]
            for vpol in range(vnpol):
                vis[..., vchan, vpol] = ng.dirty2ms(fuvw.astype(numpy.float64),
                                    freq[vchan:vchan + 1].astype(numpy.float64),
                                    model.data[imchan, vpol, :, :].T.astype(numpy.float64),
                                    pixsize_x=pixsize,
                                    pixsize_y=pixsize,
                                    epsilon=epsilon,
                                    do_wstacking=do_wstacking,
                                    nthreads=nthreads,
                                    verbosity=verbosity)[:,0]
        
        vis = convert_pol_frame(vis, model.polarisation_frame, bvis.polarisation_frame, polaxis=2)
        newbvis.data['vis'] = vis.reshape([nrows, nants, nants, vnchan, vnpol])

        # Now we can shift the visibility from the image frame to the original visibility frame
        return shift_vis_to_image(newbvis, model, tangent=True, inverse=True)
    
    
    def invert_ng(bvis: BlockVisibility, model: Image, dopsf: bool = False, normalize: bool = True,
                  **kwargs) -> (Image, numpy.ndarray):
        """ Invert using nifty-gridder module
        
        https://gitlab.mpcdf.mpg.de/ift/nifty_gridder
    
        Use the image im as a template. Do PSF in a separate call.
    
        This is at the bottom of the layering i.e. all transforms are eventually expressed in terms
        of this function. . Any shifting needed is performed here.
    
        :param bvis: BlockVisibility to be inverted
        :param im: image template (not changed)
        :param normalize: Normalize by the sum of weights (True)
        :return: (resulting image, sum of the weights for each frequency and polarization)
    
        """
        
        assert isinstance(bvis, BlockVisibility), bvis

        im = copy_image(model)

        nthreads = get_parameter(kwargs, "threads", 4)
        epsilon = get_parameter(kwargs, "epsilon", 1e-12)
        do_wstacking = get_parameter(kwargs, "do_wstacking", True)
        verbosity = get_parameter(kwargs, "verbosity", 0)
        
        sbvis = copy_visibility(bvis)
        sbvis = shift_vis_to_image(sbvis, im, tangent=True, inverse=False)

        vis = bvis.vis
        
        freq = sbvis.frequency  # frequency, Hz
        
        nrows, nants, _, vnchan, vnpol = vis.shape
        uvw = sbvis.uvw.reshape([nrows * nants * nants, 3])
        ms = vis.reshape([nrows * nants * nants, vnchan, vnpol])
        wgt = sbvis.imaging_weight.reshape([nrows * nants * nants, vnchan, vnpol])
        
        if dopsf:
            ms[...] = 1.0 + 0.0j
        
        if epsilon > 5.0e-6:
            ms = ms.astype("c8")
            wgt = wgt.astype("f4")
        
        # Find out the image size/resolution
        npixdirty = im.nwidth
        pixsize = numpy.abs(numpy.radians(im.wcs.wcs.cdelt[0]))
        
        fuvw = uvw.copy()
        # We need to flip the u and w axes.
        fuvw[:, 0] *= -1.0
        fuvw[:, 2] *= -1.0
        
        nchan, npol, ny, nx = im.shape
        im.data[...] = 0.0
        sumwt = numpy.zeros([nchan, npol])
        
        ms = convert_pol_frame(ms, bvis.polarisation_frame, im.polarisation_frame, polaxis=2)
        # There's a latent problem here with the weights.
        # wgt = numpy.real(convert_pol_frame(wgt, bvis.polarisation_frame, im.polarisation_frame, polaxis=2))

        # Set up the conversion from visibility channels to image channels
        vis_to_im = numpy.round(model.wcs.sub([4]).wcs_world2pix(freq, 0)[0]).astype('int')
        for vchan in range(vnchan):
            ichan = vis_to_im[vchan]
            for pol in range(npol):
                # Nifty gridder likes to receive contiguous arrays
                ms_1d = numpy.array([ms[row, vchan:vchan+1, pol] for row in range(nrows * nants * nants)], dtype='complex')
                ms_1d.reshape([ms_1d.shape[0], 1])
                wgt_1d = numpy.array([wgt[row, vchan:vchan+1, pol] for row in range(nrows * nants * nants)])
                wgt_1d.reshape([wgt_1d.shape[0], 1])
                dirty = ng.ms2dirty(
                    fuvw, freq[vchan:vchan+1], ms_1d, wgt_1d,
                    npixdirty, npixdirty, pixsize, pixsize, epsilon, do_wstacking=do_wstacking,
                    nthreads=nthreads, verbosity=verbosity)
                sumwt[ichan, pol] += numpy.sum(wgt[:, vchan, pol])
                im.data[ichan, pol] += dirty.T

        if normalize:
            im = normalize_sumwt(im, sumwt)


        return im, sumwt

except ImportError:
    import warnings
    
    warnings.warn('Cannot import nifty_gridder, ng disabled', ImportWarning)
    
    raise RuntimeError("Cannot import nifty_gridder")
