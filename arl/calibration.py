# Tim Cornwell <realtimcornwell@gmail.com>
#

import numpy
#
from astropy.coordinates import SkyCoord

from arl.visibility_operations import create_visibility
from arl.visibility_simulation import filter_configuration, create_named_configuration
from arl.image_operations import create_image_from_fits, replicate_image
from arl.skymodel_operations import create_skymodel_from_image
from arl.fourier_transform import predict_visibility
from arl.data_models import *

"""
Functions that either solve_gains for the calibration or apply it. On solution the gains are written into a gaintable. For
correction, the gaintable is read and, if necessary, interpolated.
"""

def solve_gains(vt: Visibility, sm: SkyModel, **kwargs) -> GainTable:
    """ Solve for calibration using a sky model
    
    :param vt:
    :type Visibility:
    :param sm:
    :type SkyModel:
    :returns: GainTable
    """
    #TODO: Implement calibration solution
    print("calibration.solve_gains: Stubbed: Solving for GainTable")
    return GainTable()


def correct_visibility(vt: Visibility, gt: GainTable, **kwargs) -> Visibility:
    """ Correct a vistable using a GainTable

    :param vt:
    :type Visibility:
    :param gt:
    :type GainTable:
    :returns: Visibility
    """
    # TODO: Implement calibration application
    print("calibration.correct_visibility: not yet implemented")
    return vt


def peel_component_visibility(vt: Visibility, sc: SkyComponent, **kwargs) -> Visibility:
    """ Correct a vistable using a GainTable

    :param vt:
    :type Visibility:
    :param sc:
    :type SkyComponent:
    :returns: Visibility, GainTable
    """
    # TODO: Implement peeling
    print("calibration.peel_component_visibility: not yet implemented")
    return vt


def aq_gaintable(gt, **kwargs):
    """Assess the quality of a gaintable

    :param im:
    :type GainTable:
    :returns: AQ
    """
    print("calibration.aq_gaintable: not yet implemented")
    return AQ()


if __name__ == '__main__':
    import os

    os.chdir('../')
    print(os.getcwd())

    kwargs = {'wstep': 100.0}

    vlaa = filter_configuration(create_named_configuration('VLAA'), **kwargs)
    vlaa.data['xyz'] *= 1.0 / 30.0
    times = numpy.arange(-3.0, +3.0, 3.0 / 60.0) * numpy.pi / 12.0
    frequency = numpy.arange(1.0e8, 1.60e8, 1e7)
    direction = SkyCoord('00h42m30s', '-41d12m00s', frame='icrs')
    vt = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=direction)
    print(vt.data)
    print(vt.frequency)
    m31image = create_image_from_fits("data/models/m31.MOD")
    print("Max, min in m31 Image = %.6f, %.6f" % (m31image.data.max(), m31image.data.min()))
    m31imagerep = replicate_image(m31image, shape=[1, 1, 1, len(frequency)])
    m31sm = create_skymodel_from_image(m31imagerep)
    vtpred = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=direction)
    vtpred = predict_visibility(vtpred, m31sm, **kwargs)
