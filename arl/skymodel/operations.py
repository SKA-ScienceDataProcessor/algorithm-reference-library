"""Function to manage skymodels.

"""

import collections
import logging
from typing import Union, List

import astropy.units as u
import numpy
from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from photutils import segmentation

from arl.data.data_models import Image, Skycomponent, assert_same_chan_pol
from arl.data.skymodel import SkyModel
from arl.calibration.operations import copy_gaintable
from arl.image.operations import copy_image
from arl.data.polarisation import PolarisationFrame
from arl.skycomponent.base import copy_skycomponent

log = logging.getLogger(__name__)


def copy_skymodel(sm):
    """ Copy a sky model
    
    """
    newsm = SkyModel()
    if sm.components is not None:
        newsm.components = [copy_skycomponent(comp) for comp in sm.components]
    if sm.images is not None:
        newsm.images = [copy_image(im) for im in sm.images]
    return newsm