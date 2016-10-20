# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

from arl.data_models import *
from arl.parameters import *
from arl.fourier_transforms import predict_visibility, invert_visibility
from arl.visibility_operations import combine_visibility

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
    vispred = predict_visibility(vis, sm, params={})
    visres = combine_visibility(vis, vispred, 1.0, -1.0)
    dirty, psf, sumwt = invert_visibility(visres, model=sm.components[0], params={})
    thresh = get_parameter(params, "threshold", 0.0)
    
    comp = sm.images[0]
    for i in range(nmajor):
        log.debug("solve_skymodel: Start of major cycle %d" % i)
        cc, res = deconvolver(dirty, psf, params={})
        comp += cc
        vispred = predict_visibility(vis, sm, params={})
        visres = combine_visibility(vis, vispred, 1.0, -1.0)
        # dirty, psf, sumwt = invert_visibility(visres, params={})
        if numpy.abs(dirty.data).max() < 1.1 * thresh:
            log.debug("Reached stopping threshold %.6f Jy" % thresh)
            break
        log.debug("solve_skymodel: End of major cycle")
    log.debug("solve_skymodel: End of major cycles")
    return visres, sm
