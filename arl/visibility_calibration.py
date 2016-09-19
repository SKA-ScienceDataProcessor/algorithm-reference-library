# Tim Cornwell <realtimcornwell@gmail.com>
#

import numpy
#
from astropy.coordinates import SkyCoord

from arl.visibility_operations import create_visibility
from arl.test_support import filter_configuration, create_named_configuration
from arl.image_operations import import_image_from_fits
from arl.skymodel_operations import create_skymodel_from_image
from arl.fourier_transforms import predict_visibility
from arl.data_models import *
from arl.parameters import get_parameter

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
    #TODO: Implement calibration solution
    log.error("visibility_calibration.solve_gains: not yet implemented")
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
    log.error("visibility_calibration.correct_visibility: not yet implemented")
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
    log.error("visibility_calibration.peel_skycomponent: not yet implemented")
    return vis


def qa_gaintable(gt, params={}):
    """Assess the quality of a gaintable

    :param im:
    :type GainTable:
    :returns: AQ
    """
    log.error("visibility_calibration.qa_gaintable: not yet implemented")
    return QA()


if __name__ == '__main__':
    import os
    from arl.image_operations import import_image_from_fits, replicate_image

    os.chdir('../')
    log.debug(os.getcwd())

    kwargs = {'wstep': 100.0}

    vlaa = filter_configuration(create_named_configuration('VLAA'), params={})
    vlaa.data['xyz'] *= 1.0 / 30.0
    times = numpy.arange(-3.0, +3.0, 3.0 / 60.0) * numpy.pi / 12.0
    frequency = numpy.arange(1.0e8, 1.60e8, 1e7)
    direction = SkyCoord('00h42m30s', '-41d12m00s', frame='icrs')
    vis = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=direction)
    log.debug(vis.data)
    log.debug(vis.frequency)
    m31image = import_image_from_fits("data/models/m31.MOD")
    log.debug("Max, min in m31 Image = %.6f, %.6f" % (m31image.data.max(), m31image.data.min()))
    m31imagerep = replicate_image(m31image, shape=[1, 1, 1, len(frequency)])
    m31sm = create_skymodel_from_image(m31imagerep)
    vispred = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=direction)
    vispred = predict_visibility(vispred, m31sm, params={})
