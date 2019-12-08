""" Visibility selectors for a BlockVisibility or Visibility.


"""
__all__ = ['vis_select_uvrange', 'vis_select_wrange']

import logging

import numpy

from data_models.memory_data_models import Visibility

log = logging.getLogger(__name__)

def vis_select_uvrange(vis: Visibility, uvmin=0.0, uvmax=numpy.infty):
    """Return rows in valid region
    
    :param vis:
    :param uvmin:
    :param uvmax:
    :return: Boolean array of valid rows
    """
    uvdist = numpy.sqrt(vis.u**2+vis.v**2)
    rows = (uvmin < uvdist) & (uvdist <= uvmax)
    return rows


def vis_select_wrange(vis: Visibility, wmax=numpy.infty):
    """Return rows in valid region

    :param vis:
    :param wmax:
    :return: Boolean array of valid rows
    """
    absw = numpy.abs(vis.w)
    rows = (wmax >= absw)
    return rows
