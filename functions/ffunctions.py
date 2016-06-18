# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of the function interface. Although the data structures are classes,
# we use stateless functions.
#
from functions.fvistable import *
from functions.fconfiguration import fconfiguration
from functions.fimage import fimage
from functions.fskymodel import fskymodel
from functions.fgaintable import fgaintable

def fsimulate_config(config: fconfiguration, **kwargs) -> fvistable:
    """
    Simulate an observation from a configuration and a skymodel
    """
    print("Simulating vistable")
    return fvistable()


def finvert_vistable(vis: fvistable, **kwargs) -> (fimage, fimage):
    """
    Invert to make dirty image and PSF
    """
    print("Inverting vistable to make dirty and psf")
    return (fimage(), fimage())


def fpredict_vistable(vis: fvistable, sm: fskymodel, **kwargs) -> fvistable:
    """
    Predict the visibility from a skymodel
    :type vis: fvistable
    """
    print("Predicting vistable from sky model")
    return fvistable()


def fcalibrate_vistable(vis: fvistable, sm: fskymodel, **kwargs) -> fgaintable:
    """
    Calibrate using a sky model
    """
    print("Solving for calibration")
    return fgaintable()

def fcorrect_vistable(vis: fvistable, gt: fgaintable, **kwargs) -> fvistable:
    """
    Correct a vistable using a gaintable
    """
    print("Applying gaintable")
    return fgaintable()


def fmajorcycles_vistable(vis: fvistable, sm: fskymodel,
                         **kwargs) -> (fvistable, fimage):
    """
    Perform major cycles
    """

    print("Performing %d major cycles" % kwargs.get('nmajor', 100))
    return vistable(), fimage()


def ffindcomponents_image(image: fimage, sm: fskymodel, **kwargs) -> fskymodel:
    """
    Find components in image
    """
    print("Finding components in image")
    return fskymodel()

if __name__ == '__main__':

    kwargs={}
    context=fcontext()

    vlaa = fconfiguration().fromname('VLAA')
    times = numpy.arange(-3.0, +3.0, 3.0 / 60.0) * numpy.pi / 12.0
    freq = numpy.arange(5e6, 150.0e6, 1e7)
    direction = SkyCoord('00h42m30s', '-41d12m00s', frame='icrs')
    vt=fvistable().observe(vlaa, times, freq, weight=1.0, direction=direction)
    print(vt)
    print(vt.freq)
    m31image=fimage().from_fits("../data/models/m31.model.fits")
    m31sm=fskymodel(m31image)
    vt=fpredict_vistable(vt, m31sm, context, **kwargs)
    dirty,psf=finvert_vistable(vt, context, **kwargs)
    sm=ffindcomponents_image(dirty, m31sm, context, **kwargs)
