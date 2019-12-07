"""
Functions that aid definition of fourier transform processing.
"""

__all__ = ['get_frequency_map', 'get_polarisation_map', 'get_rowmap']

import logging
import warnings

from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', FITSFixedWarning)

import numpy

from data_models.memory_data_models import Visibility, Image
from data_models.polarisation import PolarisationFrame

log = logging.getLogger(__name__)

def get_frequency_map(vis, im: Image = None):
    """ Map channels from visibilities to image

    """
    
    # Find the unique frequencies in the visibility
    ufrequency = numpy.unique(vis.frequency)
    vnchan = len(ufrequency)
    
    if im is None:
        spectral_mode = 'channel'
        if vis.frequency_map is None:
            vfrequencymap = get_rowmap(vis.frequency, ufrequency)
            vis.frequencymap = vfrequencymap
        else:
            vfrequencymap = vis.frequency_map
            
        assert min(vfrequencymap) >= 0, "Invalid frequency map: visibility channel < 0: %s" % str(vfrequencymap)
    
    elif im.data.shape[0] == 1 and vnchan >= 1:
        spectral_mode = 'mfs'
        if vis.frequency_map is None:
            vfrequencymap = numpy.zeros_like(vis.frequency, dtype='int')
            vis.frequencymap = vfrequencymap
        else:
            vfrequencymap = vis.frequency_map

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
    # vmap = []
    # vmap = [pdict[phash(p)] for p in col]
    # for p in col:
    #     vmap.append(pdict[phash(p)])

    n_ucol = numpy.round(col).astype(('int'))
    vmap = numpy.vectorize(pdict.__getitem__)(n_ucol)

    return vmap.tolist()
