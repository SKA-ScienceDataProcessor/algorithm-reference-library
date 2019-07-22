""" Image operations visible to the Execution Framework as Components
"""

import warnings

from astropy.wcs import FITSFixedWarning

warnings.simplefilter('ignore', FITSFixedWarning)

from data_models.memory_data_models import Image

import logging
log = logging.getLogger(__name__)

from processing_library.image.operations import create_empty_image_like

def image_gradients(im: Image):
    """Calculate image gradients numerically
    
    Gradient units are (incoming unit)/pixel e.g. Jy/beam/pixel
    
    :param im: Image
    :return: Gradient images
    """
    assert isinstance(im, Image)
    nchan, npol, ny, nx = im.shape
    
    gradientx = create_empty_image_like(im)
    gradientx.data[..., :, 1:nx] = im.data[..., :, 1:nx] - im.data[..., :, 0:(nx - 1)]
    gradienty = create_empty_image_like(im)
    gradienty.data[..., 1:ny, :] = im.data[..., 1:ny, :] - im.data[..., 0:(ny - 1), :]
    
    return gradientx, gradienty
