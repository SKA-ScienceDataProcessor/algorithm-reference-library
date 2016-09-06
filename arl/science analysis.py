# Tim Cornwell <realtimcornwell@gmail.com>
#
# Science analysis
#

import numpy

from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord

from arl.image_operations import import_image_from_fits
from arl.fourier_transforms import predict_visibility, invert_visibility, combine_visibility
from arl.data_models import *
from arl.parameters import get_parameter

def rotation_measure_synthesis(im: Image, params={}):
    """ Perform rotation measure synthesis
    
    :param im:
    :param params:
    :return:
    """
    print("science_analysis.rotation_measure_synthesis: not yet implemented")
    return im