# Tim Cornwell <realtimcornwell@gmail.com>
#
# The bulk of functions that have more than one data structure
#

import numpy
#
from astropy.coordinates import SkyCoord

from functions.visibility import visibility, simulate
from functions.configuration import configuration_filter, named_configuration
from functions.image import image_from_fits, image_replicate
from functions.skymodel import skymodel, skymodel_from_image
from functions.gaintable import gaintable
from functions.imaging import predict

def calibrate(vis: visibility, sm: skymodel, **kwargs) -> gaintable:
    """
    Calibrate using a sky model
    """
    print("calibration.calibrate: Stubbed: Solving for gaintable")
    return gaintable()


def correct(vt: visibility, gt: gaintable, **kwargs) -> visibility:
    """
    Correct a vistable using a gaintable
    """
    print("calibration.correct: Stubbed: Applying gaintab")
    return vt


if __name__ == '__main__':
    import os

    os.chdir('../')
    print(os.getcwd())

    kwargs = {'wstep':100.0}

    vlaa = configuration_filter(named_configuration('VLAA'), **kwargs)
    vlaa.data['xyz']*=1.0/30.0
    times = numpy.arange(-3.0, +3.0, 3.0 / 60.0) * numpy.pi / 12.0
    frequency = numpy.arange(1.0e8, 1.50e8, 1e7)
    direction = SkyCoord('00h42m30s', '-41d12m00s', frame='icrs')
    vt = simulate(vlaa, times, frequency, weight=1.0, direction=direction)
    print(vt.data)
    print(vt.frequency)
    m31image = image_from_fits("data/models/m31.MOD")
    print("Max, min in m31 image = %.6f, %.6f" % (m31image.data.max(), m31image.data.min()))
    m31image = image_replicate(m31image, shape=[1, 1, 1, len(frequency)])
    m31sm = skymodel_from_image(m31image)
    vtpred = simulate(vlaa, times, frequency, weight=1.0, direction=direction)
    vtpred = predict(vtpred, m31sm, **kwargs)
