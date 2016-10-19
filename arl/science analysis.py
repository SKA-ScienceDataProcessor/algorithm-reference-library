# Tim Cornwell <realtimcornwell@gmail.com>
#
# Science analysis
#

from arl.data_models import *
from arl.parameters import *

def rotation_measure_synthesis(im: Image, params=None):
    """ Perform rotation measure synthesis
    
    :param im:
    :param params: Dictionary containing parameters
    :return:
    """
    # TODO: implement

    if params is None:
        params = {}
    log.warn("rotation_measure_synthesis: not yet implemented")
    return im