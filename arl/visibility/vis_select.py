""" Visibility selectors for a BlockVisibility or Visibility.


"""

import logging

import numpy

from arl.data.data_models import Visibility
log = logging.getLogger(__name__)

def vis_select_uvrange(vis: Visibility, uvmin=0.0, uvmax=numpy.infty) -> numpy.ndarray:
    """Return rows in valid region
    
    :param vis:
    :param uvmin:
    :param uvmax:
    :return: Boolean array of valid rows
    """
    uvdist = numpy.sqrt(vis.u**2+vis.v**2)
    rows = (uvmin < uvdist) & (uvdist <= uvmax)
    return rows
