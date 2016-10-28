# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

from arl.ftprocessor import invert_2d, predict_2d

from data.data_models import *
from data.parameters import *
from visibility.visibility_operations import combine_visibility

log = logging.getLogger("arl.skymodel_solvers")

def solve_skymodel(vis: Visibility, sm: Skymodel, deconvolver, params=None):
    """Solve for Skymodel using a deconvolver. The interface of deconvolver is the same as clean.

    This is the same as a majorcycle.

    :param params:
    :param vis:
    :param sm:
    :param deconvolver: Deconvolver to be used e.g. msclean
    :arg function:
    :returns: Visibility, Skymodel
    """
    if params is None:
        params = {}
    log_parameters(params)
    nmajor = get_parameter(params, 'nmajor', 5)
    log.debug("solve_combinations.solve_skymodel: Performing %d major cycles" % nmajor)
    
    # The model is added to each major cycle and then the visibilities are
    # calculated from the full model
    vispred = predict_2d(vis, sm, params={})
    visres = combine_visibility(vis, vispred, 1.0, -1.0)
    dirty, sumwt = invert_2d(visres, sm.images[0], params={})
    psf, sumwt = invert_2d(visres, sm.images[0], dopsf=True, params={})
    thresh = get_parameter(params, "threshold", 0.0)
    
    comp = sm.images[0]
    for i in range(nmajor):
        log.debug("solve_skymodel: Start of major cycle %d" % i)
        cc, res = deconvolver(dirty, psf, params={})
        comp += cc
        vispred = predict_2d(vis, sm.images[0], params={})
        visres = combine_visibility(vis, vispred, 1.0, -1.0)
        dirty, sumwt = invert_2d(visres, sm.images[0], params={})
        if numpy.abs(dirty.data).max() < 1.1 * thresh:
            log.debug("Reached stopping threshold %.6f Jy" % thresh)
            break
        log.debug("solve_skymodel: End of major cycle")
    log.debug("solve_skymodel: End of major cycles")
    return visres, sm
