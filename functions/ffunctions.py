# Tim Cornwell <realtimcornwell@gmail.com>
#
# The bulk of functions that have more than one data structure
#

import numpy
#

from astropy.coordinates import SkyCoord, EarthLocation
from functions.fvistab import fvistab, fvistab_from_fconfig
from functions.fconfig import fconfig, fconfig_filter, fconfig_from_name
from functions.fimage import fimage, fimage_from_fits
from functions.fskymod import fskymod, fskymod_from_fimage
from functions.fgaintab import fgaintab
from functions.fcomp import fcomp


def fimage_from_fvistable(vis: fvistab, **kwargs) -> (fimage, fimage):
    """
    Invert to make dirty image and PSF
    """
    print("Stubbed: Inverting vistable to make dirty and psf")
    return (fimage(), fimage())


def fvistable_from_fskymod(vis: fvistab, sm: fskymod, **kwargs) -> fvistab:
    """
    Predict the visibility from a skymodel
    :type vis: fvistab
    """
    print("Stubbed: Predicting vistable from sky model")
    return fvistab()


def fgaintab_from_vistab(vis: fvistab, sm: fskymod, **kwargs) -> fgaintab:
    """
    Calibrate using a sky model
    """
    print("Stubbed: Solving for fgaintab")
    return fgaintab()


def fvistab_apply_gaintab(vis: fvistab, gt: fgaintab, **kwargs) -> fvistab:
    """
    Correct a vistable using a gaintable
    """
    print("Stubbed: Applying gaintab")
    return fgaintab()


def fvistab_iterate_fskymod(vis: fvistab, sm: fskymod, **kwargs) -> (fvistab, fskymod):
    """
    Perform major cycles
    """

    print("Stubbed: Performing %d major cycles" % kwargs.get('nmajor', 100))
    return vis, sm


def fcomp_from_fskymod(vis: fvistab, direction: SkyCoord, **kwargs) -> fcomp:
    """
    Direct Fourier summation
    """

    print("Stubbed: Performing Direct Fourier Summation in direction %s")
    return fcomp()


def fcomp_fit_fimage(image: fimage, sm: fskymod, **kwargs) -> (fskymod, fimage):
    """
    Find components in image, return skymodel and residual image
    """
    print("Stubbed: Finding components in image, adding to skymodel")
    return sm, image


if __name__ == '__main__':
    kwargs = {}

    vlaa = fconfig_filter(fconfig_from_name('VLAA'), **kwargs)
    times = numpy.arange(-3.0, +3.0, 3.0 / 60.0) * numpy.pi / 12.0
    freq = numpy.arange(5e6, 150.0e6, 1e7)
    direction = SkyCoord('00h42m30s', '-41d12m00s', frame='icrs')
    vt = fvistab_from_fconfig(vlaa, times, freq, weight=1.0, direction=direction)
    print(vt.data)
    print(vt.frequency)
    m31image = fimage_from_fits("../data/models/m31.model.fits")
    m31sm = fskymod_from_fimage(m31image)
    vt = fvistable_from_fskymod(vt, m31sm, **kwargs)
    dirty, psf = fimage_from_fvistable(vt, **kwargs)
    m31smnew, res = fcomp_fit_fimage(dirty, m31sm, **kwargs)
