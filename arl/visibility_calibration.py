# Tim Cornwell <realtimcornwell@gmail.com>
#

import numpy
#
from astropy.coordinates import SkyCoord

from arl.visibility_operations import create_visibility
from arl.testing_support import filter_configuration, create_named_configuration
from arl.image_operations import import_image_from_fits
from arl.skymodel_operations import create_skymodel_from_image
from arl.fourier_transforms import predict_visibility
from arl.data_models import *
from arl.parameters import *

import logging
log = logging.getLogger("arl.visibility_calibration")

"""
Functions that either solve_gains for the calibration or apply it. On solution the gains are written into a gaintable. For
correction, the gaintable is read and, if necessary, interpolated.
"""

def solve_gains(vis: Visibility, sm: SkyModel, params={}) -> GainTable:
    """ Solve for calibration using a sky model
    
    :param vis:
    :type Visibility: Visibility to be processed
    :param sm:
    :type SkyModel:
    :returns: GainTable
    """
    log_parameters(params)
    #TODO: Implement calibration solution
    log.error("solve_gains: not yet implemented")
    return GainTable()


def correct_visibility(vis: Visibility, gt: GainTable, params={}) -> Visibility:
    """ Correct a vistable using a GainTable

    :param vis: Visibility to be processed
    :type Visibility:
    :param gt: GainTable
    :type GainTable:
    :returns: Visibility
    """
    # TODO: Implement calibration application
    log_parameters(params)
    log.error("correct_visibility: not yet implemented")
    return vis


def peel_skycomponent(vis: Visibility, sc: SkyComponent, params={}) -> Visibility:
    """ Correct a vistable using a GainTable

    :param vis: Visibility to be processed
    :type Visibility:
    :param sc:
    :type SkyComponent:
    :returns: Visibility, GainTable
    """
    # TODO: Implement peeling
    log_parameters(params)
    log.error("peel_skycomponent: not yet implemented")
    return vis


def qa_gaintable(gt, params={}):
    """Assess the quality of a gaintable

    :param im:
    :type GainTable:
    :returns: AQ
    """
    # TODO: implement

    log_parameters(params)
    log.error("qa_gaintable: not yet implemented")
    return QA()
