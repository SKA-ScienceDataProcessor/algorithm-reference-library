"""
Functions that aid definition of fourier transform processing.
"""

import logging
import warnings

from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', FITSFixedWarning)

import numpy

from data_models.memory_data_models import Visibility, Image
from data_models.parameters import get_parameter
from data_models.polarisation import PolarisationFrame

from ..fourier_transforms.convolutional_gridding import anti_aliasing_calculate
from ..image.operations import convert_image_to_kernel
from ..image.operations import copy_image, fft_image, pad_image, create_w_term_like

log = logging.getLogger(__name__)


def get_frequency_map(vis, im: Image = None):
    """ Map channels from visibilities to image

    """
    
    # Find the unique frequencies in the visibility
    ufrequency = numpy.unique(vis.frequency)
    vnchan = len(ufrequency)
    
    if im is None:
        spectral_mode = 'channel'
        vfrequencymap = get_rowmap(vis.frequency, ufrequency)
        assert min(vfrequencymap) >= 0, "Invalid frequency map: visibility channel < 0: %s" % str(vfrequencymap)
    
    elif im.data.shape[0] == 1 and vnchan >= 1:
        spectral_mode = 'mfs'
        vfrequencymap = numpy.zeros_like(vis.frequency, dtype='int')
    
    else:
        # We can map these to image channels
        v2im_map = im.wcs.sub(['spectral']).wcs_world2pix(ufrequency, 0)[0].astype('int')
        
        spectral_mode = 'channel'
        nrows = len(vis.frequency)
        row2vis = numpy.array(get_rowmap(vis.frequency, ufrequency))
        vfrequencymap = [v2im_map[row2vis[row]] for row in range(nrows)]
        
        assert min(vfrequencymap) >= 0, "Invalid frequency map: image channel < 0 %s" % str(vfrequencymap)
        assert max(vfrequencymap) < im.shape[0], "Invalid frequency map: image channel > number image channels %s" % \
                                                 str(vfrequencymap)
    
    return spectral_mode, vfrequencymap


def get_polarisation_map(vis: Visibility, im: Image = None):
    """ Get the mapping of visibility polarisations to image polarisations
    
    """
    if vis.polarisation_frame == im.polarisation_frame:
        if vis.polarisation_frame == PolarisationFrame('stokesI'):
            return "stokesI->stokesI", lambda pol: 0
        elif vis.polarisation_frame == PolarisationFrame('stokesIQUV'):
            return "stokesIQUV->stokesIQUV", lambda pol: pol
    
    return "unknown", lambda pol: pol


def get_rowmap(col, ucol=None):
    """ Map to unique cols
    
    :param col: Data column
    :param ucol: Unique values in col
    """
    pdict = {}
    
    def phash(f):
        return numpy.round(f).astype('int')
    
    if ucol is None:
        ucol = numpy.unique(col)
        
    for i, f in enumerate(ucol):
        pdict[phash(f)] = i
    vmap = []
    for p in col:
        vmap.append(pdict[phash(p)])

    return vmap


def get_uvw_map(vis: Visibility, im: Image, padding=2):
    """ Get the generators that map channels uvw to pixels

    :param vis:
    :param im:
    :param padding:
    :return: uvw mode, shape, padding, uvw mapping
    """
    # Transform parameters
    
    # Model image information
    inchan, inpol, ny, nx = im.data.shape
    shape = (1, int(round(padding * ny)), int(round(padding * nx)))
    # UV sampling information
    uvwscale = numpy.zeros([3])
    uvwscale[0:2] = im.wcs.wcs.cdelt[0:2] * numpy.pi / 180.0
    assert uvwscale[0] != 0.0, "Error in uv scaling"
    
    vuvwmap = uvwscale * vis.uvw
    uvw_mode = "2d"
    
    return uvw_mode, shape, padding, vuvwmap