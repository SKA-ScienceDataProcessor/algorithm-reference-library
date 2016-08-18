# Tim Cornwell <realtimcornwell@gmail.com>
#
# The imaging arl that have more than one data structure
#

import numpy

from arl.define_skymodel import SkyModel
from arl.fourier_transform import predict_visibility, invert_visibility
from arl.define_visibility import Visibility, combine_visibility, GainTable

"""
Functions that solve for combinations e.g. major cycle
"""

def solve_skymodel(vt: Visibility, sm: SkyModel, deconvolver, **kwargs):
    """Solve for SkyModel a deconvolver. The interface of deconvolver is the same as clean.
    
    This is the same as a majorcycle.
    
    :param vt:
    :type Visibility:
    :param sm:
    :type SkyModel:
    :param deconvolver: Deconvolver to be used e.g. msclean
    :arg function:
    :returns: Visibility, SkyModel
    """
    nmajor = kwargs.get('nmajor', 5)
    print("solve_combinations.solve_skymodel: Performing %d major cycles" % nmajor)
    
    # The model is added to each major cycle and then the visibilities are
    # calculated from the full model
    vtpred = predict_visibility(vt, sm, **kwargs)
    vtres = combine_visibility(vt, vtpred, 1.0, -1.0)
    dirty, psf, sumwt = invert_visibility(vtres, **kwargs)
    thresh = kwargs.get("threshold", 0.0)
    
    comp = sm.images[0]
    for i in range(nmajor):
        print("solve_combinations.solve_skymodel: Start of major cycle %d" % i)
        cc, res = deconvolver(dirty, psf, **kwargs)
        comp += cc
        vtpred = predict_visibility(vt, sm, **kwargs)
        vtres = combine_visibility(vt, vtpred, 1.0, -1.0)
        dirty, psf, sumwt = invert_visibility(vtres, **kwargs)
        if numpy.abs(dirty.data).max() < 1.1 * thresh:
            print("Reached stopping threshold %.6f Jy" % thresh)
            break
        print("solve_combinations.solve_skymodel: End of major cycle")
    print("solve_combinations.solve_skymodel: End of major cycles")
    return vtres, sm


def solve_skymodel_gains(vt: Visibility, sm: SkyModel, deconvolver, **kwargs):
    """Solve for SkyModel a deconvolver. The interface of deconvolver is the same as clean.

    This is the same as self-calibration

    :param vt:
    :type Visibility:
    :param sm:
    :type SkyModel:
    :param deconvolver: Deconvolver to be used e.g. msclean
    :arg function:
    :returns: Visibility, SkyModel, Gaintable
    """
    print("solve_combinations.solve_skymodel_gains: not implemeneted yet")
    return vt, sm, GainTable
