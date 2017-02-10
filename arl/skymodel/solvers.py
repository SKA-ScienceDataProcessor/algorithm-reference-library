# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Definition of structures needed by the function interface. These are mostly
subclasses of astropy classes.
"""

import copy
from arl.fourier_transforms.ftprocessor import invert_2d_base, predict_2d_base

from arl.data.data_models import *
from arl.data.parameters import *
from arl.visibility.operations import copy_visibility

log = logging.getLogger(__name__)

def solve_skymodel(vis: CompressedVisibility, sm: Skymodel, deconvolver, **kwargs):
    """Solve for Skymodel using a deconvolver. The interface of deconvolver is the same as clean.

    This is the same as a majorcycle.

    :param params:
    :param vis:
    :param sm:
    :param deconvolver: Deconvolver to be used e.g. msclean
    :arg function:
    :returns: CompressedVisibility, Skymodel
    """
    nmajor = get_parameter(kwargs, 'nmajor', 5)
    log.info("solve_combinations.solve_skymodel: Performing %d major cycles" % nmajor)
    
    # The model is added to each major cycle and then the visibilities are
    # calculated from the full model
    vispred = predict_2d_base(vis, sm, **kwargs)
    visres = copy_visibility(vispred)
    visres.data['vis'] = vis.data['vis'] - vispred.data['vis']
    dirty, sumwt = invert_2d_base(visres, sm.images[0], **kwargs)
    psf, sumwt = invert_2d_base(visres, sm.images[0], dopsf=True, **kwargs)
    thresh = get_parameter(kwargs, "threshold", 0.0)
    
    comp = sm.images[0]
    for i in range(nmajor):
        log.info("solve_skymodel: Start of major cycle %d" % i)
        cc, res = deconvolver(dirty, psf, **kwargs)
        comp += cc
        vispred = predict_2d_base(vis, sm.images[0], **kwargs)
        visres.data['vis'] = vis.data['vis'] - vispred.data['vis']
        dirty, sumwt = invert_2d_base(visres, sm.images[0], **kwargs)
        if numpy.abs(dirty.data).max() < 1.1 * thresh:
            log.info("Reached stopping threshold %.6f Jy" % thresh)
            break
        log.info("solve_skymodel: End of major cycle")
    log.info("solve_skymodel: End of major cycles")
    return visres, sm
