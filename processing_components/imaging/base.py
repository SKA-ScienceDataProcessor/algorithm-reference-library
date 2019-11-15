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

import collections
import logging
from typing import List, Union, Tuple

import numpy
import nifty_gridder as ng
from astropy import constants as constants
from astropy import units as units
from astropy import wcs
from astropy.wcs.utils import pixel_to_skycoord

from data_models.memory_data_models import Visibility, BlockVisibility, Image, Skycomponent, assert_same_chan_pol
from data_models.parameters import get_parameter
from data_models.polarisation import convert_pol_frame, PolarisationFrame

from processing_library.image.operations import create_image_from_array
from processing_library.imaging.imaging_params import get_frequency_map
from processing_library.util.coordinate_support import simulate_point, skycoord_to_lmn

from processing_components.griddata.kernels  import create_pswf_convolutionfunction
from ..griddata.gridding import grid_visibility_to_griddata, \
    fft_griddata_to_image, fft_image_to_griddata, \
    degrid_visibility_from_griddata
from ..griddata.operations import create_griddata_from_image
from ..visibility.base import copy_visibility, phaserotate_visibility

log = logging.getLogger(__name__)


def shift_vis_to_image(vis: Visibility, im: Image, tangent: bool = True, inverse: bool = False) \
        -> Visibility:
    """Shift visibility to the FFT phase centre of the image in place

    :param vis: Visibility data
    :param im: Image model used to determine phase centre
    :param tangent: Is the shift purely on the tangent plane True|False
    :param inverse: Do the inverse operation True|False
    :return: visibility with phase shift applied and phasecentre updated

    """
    assert isinstance(vis, Visibility), "vis is not a Visibility: %r" % vis
    
    nchan, npol, ny, nx = im.data.shape
    
    # Convert the FFT definition of the phase center to world coordinates (1 relative)
    # This is the only place in ARL where the relationship between the image and visibility
    # frames is defined.
    
    image_phasecentre = pixel_to_skycoord(nx // 2 + 1, ny // 2 + 1, im.wcs, origin=1)
    if vis.phasecentre.separation(image_phasecentre).rad > 1e-15:
        if inverse:
            log.debug("shift_vis_from_image: shifting phasecentre from image phase centre %s to visibility phasecentre "
                      "%s" % (image_phasecentre, vis.phasecentre))
        else:
            log.debug("shift_vis_from_image: shifting phasecentre from vis phasecentre %s to image phasecentre %s" %
                      (vis.phasecentre, image_phasecentre))
        vis = phaserotate_visibility(vis, image_phasecentre, tangent=tangent, inverse=inverse)
        vis.phasecentre = im.phasecentre
    
    assert isinstance(vis, Visibility), "after phase_rotation, vis is not a Visibility"
    
    return vis


def normalize_sumwt(im: Image, sumwt) -> Image:
    """Normalize out the sum of weights

    :param im: Image, im.data has shape [nchan, npol, ny, nx]
    :param sumwt: Sum of weights [nchan, npol]
    """
    nchan, npol, _, _ = im.data.shape
    assert isinstance(im, Image), im
    assert sumwt is not None
    assert nchan == sumwt.shape[0]
    assert npol == sumwt.shape[1]
    for chan in range(nchan):
        for pol in range(npol):
            if sumwt[chan, pol] > 0.0:
                im.data[chan, pol, :, :] = im.data[chan, pol, :, :] / sumwt[chan, pol]
            else:
                im.data[chan, pol, :, :] = 0.0
    return im


def predict_2d(vis: Union[BlockVisibility, Visibility], model: Image, gcfcf=None,
               **kwargs) -> Union[BlockVisibility, Visibility]:
    """ Predict using convolutional degridding.

    This is at the bottom of the layering i.e. all transforms are eventually expressed in terms of
    this function. Any shifting needed is performed here.

    :param vis: Visibility to be predicted
    :param model: model image
    :param gcfcf: (Grid correction function i.e. in image space, Convolution function i.e. in uv space)
    :return: resulting visibility (in place works)
    """
    
    if model is None:
        return vis
    
    assert isinstance(vis, Visibility), vis

    _, _, ny, nx = model.data.shape
    
    if gcfcf is None:
        gcf, cf = create_pswf_convolutionfunction(model,
                                                  support=get_parameter(kwargs, "support", 6),
                                                  oversampling=get_parameter(kwargs, "oversampling", 128))
    else:
        gcf, cf = gcfcf
    
    griddata = create_griddata_from_image(model)
    griddata = fft_image_to_griddata(model, griddata, gcf)
    vis = degrid_visibility_from_griddata(vis, griddata=griddata, cf=cf)
    
    # Now we can shift the visibility from the image frame to the original visibility frame
    svis = shift_vis_to_image(vis, model, tangent=True, inverse=True)
    
    return svis

def predict_ng(bvis: Union[BlockVisibility, Visibility], model: Image, gcfcf=None, nthreads=4,
                epsilon=6.0e-6, **kwargs) -> Union[BlockVisibility, Visibility]:
    """ Predict using convolutional degridding.
    Nifty-gridder version.

    :param bvis: BlockVisibility to be predicted
    :param model: model image
    :param nthreads: OpenMP threads number
    :param epsilon: a level of tolerance
 
    :return: resulting BlockVisibility (in place works)
    """
    
    if model is None:
        return bvis
    
    assert isinstance(bvis, BlockVisibility), bvis

    newbvis = copy_visibility(bvis, zero=True)

    # Extracting data from BlockVisibility
    freq = bvis.frequency                         #frequency, Hz
    uvw_nonzero = numpy.nonzero(bvis.uvw[:,:,:,0])
    uvw = bvis.uvw[uvw_nonzero]                   # UVW, meters [:,3]
    ms = bvis.vis[uvw_nonzero]                    # Visibility data [:,nfreq,npol]
    ms[:,:,:] = 0.0 + 0.0j                        # Make all vis data equal to 0 +0j 
    wgt = numpy.ones((ms.shape[0],ms.shape[2]))      # All weights equal to 1.0
    v_nchan = ms.shape[1]
    v_npol = ms.shape[2]
    
    # Get the image properties
    m_nchan, m_npol, ny, nx = model.data.shape
    #print(m_nchan, v_nchan, m_npol, v_npol, nx,ny)
    # Check if the number of frequency channels matches in bvis and a model
    assert(m_nchan == v_nchan)
    assert(m_npol == v_npol)
    
    # Set parameters for ng.dirty2ms()
    do_wstacking=True
    # Find out the image size/resolution
    pixsize = numpy.abs(numpy.radians(model.wcs.wcs.cdelt[0]))
    wgtt = None
    
    bvis.data['vis'][...] = 0.0 + 0.0j
    # Make de-gridding over a frequency range and pol fields
    for i in range(v_nchan):
        for j in range(v_npol):
            ngvis = ng.dirty2ms(uvw.astype(numpy.float64),
                            freq[i:i+1].astype(numpy.float64),
                            model.data[i,j,:,:], 
                            wgtt,
                            pixsize, 
                            pixsize, 
                            epsilon,
                            do_wstacking=do_wstacking, 
                            nthreads=nthreads, 
                            verbosity=2)
            #print(i, j, len(ngvis), ngvis.shape)
        # re-write ngvis into bvis
            iflat = 0
            nants = bvis.data["uvw"].shape[1]
            ntimes = bvis.uvw.shape[0]
            for it in range(ntimes):
                for iant2 in range(nants):
                    for iant1 in range(iant2+1, nants):
                        newbvis.data['vis'][it,iant2,iant1,i,j] = ngvis[iflat]
                        newbvis.data['vis'][it,iant1,iant2,i,j] = numpy.conjugate(ngvis[iflat])
                        #print(iflat, bvis.data["uvw"][it,iant1,iant2], uvw[iflat])
                        iflat += 1
                    # if iflat + nants < len(ngvis):
                    #     newbvis.data['vis'][it, iant2, range(nants), i, j] = ngvis[iflat:(iflat+nants)][:,0]
                    # iflat += nants
                            
    # Now we can shift the visibility from the image frame to the original visibility frame
    #sbvis = shift_vis_to_image(bvis, model, tangent=True, inverse=True)

    return newbvis

def invert_2d(vis: Visibility, im: Image, dopsf: bool = False, normalize: bool = True,
              gcfcf=None, **kwargs) -> (Image, numpy.ndarray):
    """ Invert using 2D convolution function, using the specified convolution function

    Use the image im as a template. Do PSF in a separate call.

    This is at the bottom of the layering i.e. all transforms are eventually expressed in terms
    of this function. . Any shifting needed is performed here.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    :param gcfcf: (Grid correction function i.e. in image space, Convolution function i.e. in uv space)
    :return: resulting image

    """
    assert isinstance(vis, Visibility), vis
    
    svis = copy_visibility(vis)
    
    if dopsf:
        svis.data['vis'][...] = 1.0+0.0j
    
    svis = shift_vis_to_image(svis, im, tangent=True, inverse=False)

    if gcfcf is None:
        gcf, cf = create_pswf_convolutionfunction(im,
                                                  support=get_parameter(kwargs, "support", 6),
                                                  oversampling=get_parameter(kwargs, "oversampling", 128))
    else:
        gcf, cf = gcfcf

    griddata = create_griddata_from_image(im)
    griddata, sumwt = grid_visibility_to_griddata(svis, griddata=griddata, cf=cf)
    
    imaginary = get_parameter(kwargs, "imaginary", False)
    if imaginary:
        result0, result1 = fft_griddata_to_image(griddata, gcf, imaginary=imaginary)
        log.debug("invert_2d: retaining imaginary part of dirty image")
        if normalize:
            result0 = normalize_sumwt(result0, sumwt)
            result1 = normalize_sumwt(result1, sumwt)
        return result0, sumwt, result1
    else:
        result = fft_griddata_to_image(griddata, gcf)
        if normalize:
            result = normalize_sumwt(result, sumwt)
        return result, sumwt

def invert_ng(bvis: BlockVisibility, im: Image, dopsf: bool = False, normalize: bool = True,
              dataCube: bool = True, nthreads=4, epsilon=6.0e-6, **kwargs)-> (Image, numpy.ndarray):
    """ Invert using nifty-gridder module

    Use the image im as a template. Do PSF in a separate call.

    This is at the bottom of the layering i.e. all transforms are eventually expressed in terms
    of this function. . Any shifting needed is performed here.

    :param bvis: BlockVisibility to be inverted
    :param im: image template (not changed)
    :param normalize: Normalize by the sum of weights (True)
    :param dataCube: make inversion for each frequency
    :param nthreads: OpenMP threads number
    :param epsilon: a level of tolerance
    :return: resulting image
    :return: sum of the weights for each frequency and polarization

    """
    
    assert isinstance(bvis, BlockVisibility), bvis

    sbvis = copy_visibility(bvis)

    if dopsf:
        sbvis.data['vis'][...] = 1.0 + 0.0j

    #sbvis = shift_vis_to_image(sbvis, im, tangent=True, inverse=False)

    # Extracting data from BlockVisibility
    freq = sbvis.frequency                         #frequency, Hz
    uvw_nonzero = numpy.nonzero(sbvis.uvw[:,:,:,0])
    uvw = sbvis.uvw[uvw_nonzero]                   # UVW, meters [:,3]
    ms = sbvis.vis[uvw_nonzero]                    # Visibility data [:,nfreq,npol]
    wgt = numpy.ones((ms.shape[0],ms.shape[2]))      # All weights equal to 1.0
    # Add up XX and YY if polarized data
    if ms.shape[2] == 1: # Scalar
        idx = [0]        # Only I
    else:                # Polar
        idx = [0,3]      # XX and YY
    ms = numpy.sum(ms[:,:,idx],axis=2)
    wgt = 1/numpy.sum(1/wgt, axis=1)
    
    # Assing the weights to all frequencies
    wgt = numpy.repeat(wgt[:,None], len(freq),axis=1)
    print(wgt.shape)
    do_wstacking=True
    if epsilon > 5.0e-6:
        ms = ms.astype("c8")
        wgt = wgt.astype("f4")
    
    # Find out the image size/resolution
    npixdirty = im.nwidth
    pixsize = numpy.abs(numpy.radians(im.wcs.wcs.cdelt[0]))
    
    # If non-spectral image
    if im.nchan == 1:
        dataCube = False
        # Else check if the number of frequencies in the image and MS match
    else:
        assert(im.nchan == len(freq))

    sumwt = numpy.ones((im.nchan, im.npol))
    fuvw = uvw.copy()
    # We need to flip the u and w axes.
    fuvw[:,0] *= -1.0
    fuvw[:,2] *= -1.0
    if not dataCube:
        dirty = ng.ms2dirty(
           fuvw, freq, ms, wgt, npixdirty, npixdirty, pixsize, pixsize, epsilon,
           do_wstacking=do_wstacking, nthreads=nthreads, verbosity=2)
        sumwt[0,0] = numpy.sum(wgt)
        if normalize:
            dirty = dirty/sumwt[0,0]
        im.data[0][0] = dirty.T
    else:
        for i in range(len(freq)):
            print(i, freq[i], freq[i:i+1].shape, ms[:,i:i+1].shape, wgt[:,i:i+1].shape )
            dirty = ng.ms2dirty(
              fuvw, freq[i:i+1], ms[:,i:i+1], wgt[:,i:i+1], npixdirty, npixdirty, pixsize, pixsize, epsilon,
              do_wstacking=do_wstacking, nthreads=nthreads, verbosity=2)
            sumwt[i,0] = numpy.sum(wgt[:,i:i+1])
            if normalize:
                dirty = dirty/sumwt[i,0]
            im.data[i][0] = dirty.T
    
    return im, sumwt



def predict_skycomponent_visibility(vis: Union[Visibility, BlockVisibility],
                                    sc: Union[Skycomponent, List[Skycomponent]]) -> Union[Visibility, BlockVisibility]:
    """Predict the visibility from a Skycomponent, add to existing visibility, for Visibility or BlockVisibility

    :param vis: Visibility or BlockVisibility
    :param sc: Skycomponent or list of SkyComponents
    :return: Visibility or BlockVisibility
    """
    if sc is None:
        return vis
    
    if not isinstance(sc, collections.Iterable):
        sc = [sc]

    
    if isinstance(vis, Visibility):
        
        _, im_nchan = list(get_frequency_map(vis, None))
        
        for comp in sc:
            assert isinstance(comp, Skycomponent), comp
            
            assert_same_chan_pol(vis, comp)
            
            l, m, n = skycoord_to_lmn(comp.direction, vis.phasecentre)
            phasor = simulate_point(vis.uvw, l, m)
            
            comp_flux = comp.flux[im_nchan, :]
            vis.data['vis'][...] += comp_flux[:,:] * phasor[:, numpy.newaxis]

    elif isinstance(vis, BlockVisibility):
        
        ntimes, nant, _, nchan, npol = vis.vis.shape
        
        k = numpy.array(vis.frequency) / constants.c.to('m s^-1').value
        
        for comp in sc:
            #            assert isinstance(comp, Skycomponent), comp
            assert_same_chan_pol(vis, comp)
            
            flux = comp.flux
            if comp.polarisation_frame != vis.polarisation_frame:
                flux = convert_pol_frame(flux, comp.polarisation_frame, vis.polarisation_frame)
            
            l, m, n = skycoord_to_lmn(comp.direction, vis.phasecentre)
            uvw = vis.uvw[..., numpy.newaxis] * k
            phasor = numpy.ones([ntimes, nant, nant, nchan, npol], dtype='complex')
            for chan in range(nchan):
                phasor[:, :, :, chan, :] = simulate_point(uvw[..., chan], l, m)[..., numpy.newaxis]
            
            vis.data['vis'][..., :, :] += flux[:, :] * phasor[..., :]
    
    return vis


def create_image_from_visibility(vis, **kwargs) -> Image:
    """Make an empty image from params and Visibility
    
    This makes an empty, template image consistent with the visibility, allowing optional overriding of select
    parameters. This is a convenience function and does not transform the visibilities.

    :param vis:
    :param phasecentre: Phasecentre (Skycoord)
    :param channel_bandwidth: Channel width (Hz)
    :param cellsize: Cellsize (radians)
    :param npixel: Number of pixels on each axis (512)
    :param frame: Coordinate frame for WCS (ICRS)
    :param equinox: Equinox for WCS (2000.0)
    :param nchan: Number of image channels (Default is 1 -> MFS)
    :return: image
    """
    assert isinstance(vis, Visibility) or isinstance(vis, BlockVisibility), \
        "vis is not a Visibility or a BlockVisibility: %r" % (vis)
    
    log.debug("create_image_from_visibility: Parsing parameters to get definition of WCS")
    
    imagecentre = get_parameter(kwargs, "imagecentre", vis.phasecentre)
    phasecentre = get_parameter(kwargs, "phasecentre", vis.phasecentre)
    
    # Spectral processing options
    ufrequency = numpy.unique(vis.frequency)
    vnchan = len(ufrequency)
    
    frequency = get_parameter(kwargs, "frequency", vis.frequency)
    inchan = get_parameter(kwargs, "nchan", vnchan)
    reffrequency = frequency[0] * units.Hz
    channel_bandwidth = get_parameter(kwargs, "channel_bandwidth", 0.99999999999 * vis.channel_bandwidth[0]) * units.Hz
    
    if (inchan == vnchan) and vnchan > 1:
        log.debug(
            "create_image_from_visibility: Defining %d channel Image at %s, starting frequency %s, and bandwidth %s"
            % (inchan, imagecentre, reffrequency, channel_bandwidth))
    elif (inchan == 1) and vnchan > 1:
        assert numpy.abs(channel_bandwidth.value) > 0.0, "Channel width must be non-zero for mfs mode"
        log.debug("create_image_from_visibility: Defining single channel MFS Image at %s, starting frequency %s, "
                 "and bandwidth %s"
                 % (imagecentre, reffrequency, channel_bandwidth))
    elif inchan > 1 and vnchan > 1:
        assert numpy.abs(channel_bandwidth.value) > 0.0, "Channel width must be non-zero for mfs mode"
        log.debug("create_image_from_visibility: Defining multi-channel MFS Image at %s, starting frequency %s, "
                 "and bandwidth %s"
                 % (imagecentre, reffrequency, channel_bandwidth))
    elif (inchan == 1) and (vnchan == 1):
        assert numpy.abs(channel_bandwidth.value) > 0.0, "Channel width must be non-zero for mfs mode"
        log.debug("create_image_from_visibility: Defining single channel Image at %s, starting frequency %s, "
                 "and bandwidth %s"
                 % (imagecentre, reffrequency, channel_bandwidth))
    else:
        raise ValueError("create_image_from_visibility: unknown spectral mode ")
    
    # Image sampling options
    npixel = get_parameter(kwargs, "npixel", 512)
    uvmax = numpy.max((numpy.abs(vis.data['uvw'][:, 0:1])))
    if isinstance(vis, BlockVisibility):
        uvmax *= numpy.max(frequency) / constants.c.to('m s^-1').value
    log.debug("create_image_from_visibility: uvmax = %f wavelengths" % uvmax)
    criticalcellsize = 1.0 / (uvmax * 2.0)
    log.debug("create_image_from_visibility: Critical cellsize = %f radians, %f degrees" % (
        criticalcellsize, criticalcellsize * 180.0 / numpy.pi))
    cellsize = get_parameter(kwargs, "cellsize", 0.5 * criticalcellsize)
    log.debug("create_image_from_visibility: Cellsize          = %g radians, %g degrees" % (cellsize,
                                                                                           cellsize * 180.0 / numpy.pi))
    override_cellsize = get_parameter(kwargs, "override_cellsize", True)
    if override_cellsize and cellsize > criticalcellsize:
        log.debug("create_image_from_visibility: Resetting cellsize %g radians to criticalcellsize %g radians" % (
            cellsize, criticalcellsize))
        cellsize = criticalcellsize
    pol_frame = get_parameter(kwargs, "polarisation_frame", PolarisationFrame("stokesI"))
    inpol = pol_frame.npol
    
    # Now we can define the WCS, which is a convenient place to hold the info above
    # Beware of python indexing order! wcs and the array have opposite ordering
    shape = [inchan, inpol, npixel, npixel]
    log.debug("create_image_from_visibility: image shape is %s" % str(shape))
    w = wcs.WCS(naxis=4)
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, 1.0, channel_bandwidth.to(units.Hz).value]
    # The numpy definition of the phase centre of an FFT is n // 2 (0 - rel) so that's what we use for
    # the reference pixel. We have to use 0 rel everywhere.
    w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 1.0, reffrequency.to(units.Hz).value]
    w.naxis = 4
    
    # TODO: Why is this check being done?
    # direction_centre = pixel_to_skycoord(npixel // 2 + 1, npixel // 2 + 1, wcs=w, origin=1)
    # assert direction_centre.separation(imagecentre).value < 1e-7, \
    #     "Image phase centre [npixel//2, npixel//2] should be %s, actually is %s" % \
    #     (str(imagecentre), str(direction_centre))
    
    w.wcs.radesys = get_parameter(kwargs, 'frame', 'ICRS')
    w.wcs.equinox = get_parameter(kwargs, 'equinox', 2000.0)
    
    return create_image_from_array(numpy.zeros(shape), wcs=w, polarisation_frame=pol_frame)


def advise_wide_field(vis: Visibility, delA=0.02, oversampling_synthesised_beam=3.0, guard_band_image=6.0, facets=1,
                      wprojection_planes=1, verbose=True):
    """ Advise on parameters for wide field imaging.
    
    Calculate sampling requirements on various parameters
    
    For example::
    
        advice = advise_wide_field(vis, delA)
        wstep = get_parameter(kwargs, 'wstep', advice['w_sampling_primary_beam'])

    
    :param vis:
    :param delA: Allowed coherence loss (def: 0.02)
    :param oversampling_synthesised_beam: Oversampling of the synthesized beam (def: 3.0)
    :param guard_band_image: Number of primary beam half-widths-to-half-maximum to image (def: 6)
    :param facets: Number of facets on each axis
    :param wprojection_planes: Number of planes in wprojection
    :return: dict of advice
    """
    
    assert isinstance(vis, Visibility), vis
    
    max_wavelength = constants.c.to('m s^-1').value / numpy.min(vis.frequency)
    if verbose:
        log.info("advise_wide_field: Maximum wavelength %.3f (meters)" % (max_wavelength))
    
    min_wavelength = constants.c.to('m s^-1').value / numpy.max(vis.frequency)
    if verbose:
        log.info("advise_wide_field: Minimum wavelength %.3f (meters)" % (min_wavelength))

    maximum_baseline = numpy.max(numpy.abs(vis.uvw))  # Wavelengths
    if verbose:
        log.info("advise_wide_field: Maximum baseline %.1f (wavelengths)" % (maximum_baseline))
    assert maximum_baseline > 0.0, "Error in UVW coordinates: all uvw are zero"

    maximum_w = numpy.max(numpy.abs(vis.w))  # Wavelengths
    if verbose:
        log.info("advise_wide_field: Maximum w %.1f (wavelengths)" % (maximum_w))

    diameter = numpy.min(vis.configuration.diameter)
    if verbose:
        log.info("advise_wide_field: Station/dish diameter %.1f (meters)" % (diameter))
    assert diameter > 0.0, "Station/dish diameter must be greater than zero"

    primary_beam_fov = max_wavelength / diameter
    if verbose:
        log.info("advise_wide_field: Primary beam %s" % (rad_deg_arcsec(primary_beam_fov)))
    
    image_fov = primary_beam_fov * guard_band_image
    if verbose:
        log.info("advise_wide_field: Image field of view %s" % (rad_deg_arcsec(image_fov)))
    
    facet_fov = primary_beam_fov * guard_band_image / facets
    if facets > 1:
        if verbose:
            log.info("advise_wide_field: Facet field of view %s" % (rad_deg_arcsec(facet_fov)))
    
    synthesized_beam = 1.0 / (maximum_baseline)
    if verbose:
        log.info("advise_wide_field: Synthesized beam %s" % (rad_deg_arcsec(synthesized_beam)))
    
    cellsize = synthesized_beam / oversampling_synthesised_beam
    if verbose:
        log.info("advise_wide_field: Cellsize %s" % (rad_deg_arcsec(cellsize)))

    def pwr2(n):
        ex = numpy.ceil(numpy.log(n) / numpy.log(2.0)).astype('int')
        best = numpy.power(2, ex)
        return best

    def pwr23(n):
        ex = numpy.ceil(numpy.log(n) / numpy.log(2.0)).astype('int')
        best = numpy.power(2, ex)
        if best * 3 // 4 >= n:
            best = best * 3 // 4
        return best

    def pwr2345(n):
        # If pyfftw has been installed, next_fast_len would return the len of best performance
        try:
            import pyfftw
            best = pyfftw.next_fast_len(n)
        except ImportError:
            number = numpy.array([2, 3, 4, 5])
            ex = numpy.ceil(numpy.log(n) / numpy.log(number)).astype('int')
            best = min(numpy.power(number[:], ex[:]))
        return best

    npixels = int(round(image_fov / cellsize))
    if verbose:
        log.info("advice_wide_field: Npixels per side = %d" % (npixels))
    
    npixels2 = pwr2(npixels)
    if verbose:
        log.info("advice_wide_field: Npixels (power of 2) per side = %d" % (npixels2))

    npixels23 = pwr23(npixels)
    if verbose:
        log.info("advice_wide_field: Npixels (power of 2, 3) per side = %d" % (npixels23))

    npixels_min = pwr2345(npixels)
    log.info("advice_wide_field: Npixels (power of 2, 3, 4, 5) per side = %d" % (npixels_min))

    # Following equation is from Cornwell, Humphreys, and Voronkov (2012) (equation 24)
    # We will assume that the constraint holds at one quarter the entire FOV i.e. that
    # the full field of view includes the entire primary beam
    
    w_sampling_image = numpy.sqrt(2.0 * delA) / (numpy.pi * image_fov ** 2)
    if verbose:
        log.info("advice_wide_field: W sampling for full image = %.1f (wavelengths)" % (w_sampling_image))
    
    if facets > 1:
        w_sampling_facet = numpy.sqrt(2.0 * delA) / (numpy.pi * facet_fov ** 2)
        if verbose:
            log.info("advice_wide_field: W sampling for facet = %.1f (wavelengths)" % (w_sampling_facet))
    else:
        w_sampling_facet = w_sampling_image
    
    w_sampling_primary_beam = numpy.sqrt(2.0 * delA) / (numpy.pi * primary_beam_fov ** 2)
    if verbose:
        log.info("advice_wide_field: W sampling for primary beam = %.1f (wavelengths)" % (w_sampling_primary_beam))
    
    time_sampling_image = 86400.0 * w_sampling_image / (numpy.pi * maximum_baseline)
    if verbose:
        log.info("advice_wide_field: Time sampling for full image = %.1f (s)" % (time_sampling_image))
    
    if facets > 1:
        time_sampling_facet = 86400.0 * w_sampling_facet / (numpy.pi * maximum_baseline)
        if verbose:
            log.info("advice_wide_field: Time sampling for facet = %.1f (s)" % (time_sampling_facet))
    
    time_sampling_primary_beam = 86400.0 * w_sampling_primary_beam / (numpy.pi * maximum_baseline)
    if verbose:
        log.info("advice_wide_field: Time sampling for primary beam = %.1f (s)" % (time_sampling_primary_beam))
    
    freq_sampling_image = numpy.max(vis.frequency) * w_sampling_image / (numpy.pi * maximum_baseline)
    if verbose:
        log.info("advice_wide_field: Frequency sampling for full image = %.1f (Hz)" % (freq_sampling_image))
    
    if facets > 1:
        freq_sampling_facet = numpy.max(vis.frequency) * w_sampling_facet / (numpy.pi * maximum_baseline)
        if verbose:
            log.info("advice_wide_field: Frequency sampling for facet = %.1f (Hz)" % (freq_sampling_facet))
    
    freq_sampling_primary_beam = numpy.max(vis.frequency) * w_sampling_primary_beam / (numpy.pi * maximum_baseline)
    if verbose:
        log.info("advice_wide_field: Frequency sampling for primary beam = %.1f (Hz)" % (freq_sampling_primary_beam))
    
    wstep = w_sampling_primary_beam
    vis_slices = max(1, int(2 * maximum_w / wstep))
    wprojection_planes = vis_slices
    if verbose:
        log.info('advice_wide_field: Number of planes in w stack %d (primary beam)' % (vis_slices))
        log.info('advice_wide_field: Number of planes in w projection %d (primary beam)' % (wprojection_planes))

    nwpixels = int(2.0 * wprojection_planes * primary_beam_fov)
    nwpixels = nwpixels - nwpixels % 2
    if verbose:
        log.info('advice_wide_field: W support = %d (pixels) (primary beam)' % nwpixels)
    
    del pwr2
    del pwr23
    return locals()


def rad_deg_arcsec(x):
    """ Stringify x in radian and degress forms
    
    """
    return "%.3g (rad) %.3g (deg) %.3g (asec)" % (x, 180.0 * x / numpy.pi, 3600.0 * 180.0 * x / numpy.pi)
