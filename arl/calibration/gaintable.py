# Tim Cornwell <realtimcornwell@gmail.com>
#
""" Visibility operations

"""

import copy
from arl.data.data_models import GainTable, BlockVisibility
from arl.fourier_transforms.ftprocessor_params import *
from arl.visibility.operations import create_blockvisibility_from_rows
from arl.visibility.iterators import vis_timeslice_iter
from arl.util.coordinate_support import *

log = logging.getLogger(__name__)


def gaintable_summary(gt):
    """Return string summarizing the Gaintable

    """
    return "%s rows, %.3f GB" % (gt.data.shape, gt.size())

def create_gaintable_from_blockvisibility(vis: BlockVisibility, time_width: float=None, frequency_width: float=None,
                                     **kwargs):
    """ Create gain table from visibility
    
    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis

    nants = vis.nants
    utimes = numpy.unique(vis.time)
    ntimes = len(utimes)
    ufrequency = numpy.unique(vis.frequency)
    nfrequency = len(ufrequency)

    npolarisation = vis.polarisation_frame.npol

    gainshape = [ntimes, nants, nfrequency, npolarisation]
    gain = numpy.ones(gainshape, dtype='complex')
    gain_weight = numpy.ones(gainshape)
    gain_time = utimes
    gain_frequency = ufrequency
    gain_antenna = range(nants)
    
    gt = GainTable(gain=gain, time=gain_time, antenna=gain_antenna, weight=gain_weight, frequency=gain_frequency,
                   polarisation_frame=vis.polarisation_frame)
    
    return gt

def apply_gaintable(vis: BlockVisibility, gt: GainTable, **kwargs):
    """Apply a gain table to a block visibility
    
    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis

    if vis.polarisation_frame.type == Polarisation_Frame('stokesI').type:
        for chunk, rows in enumerate(vis_timeslice_iter(vis)):
            visslice = create_blockvisibility_from_rows(vis, rows)
            vistime = numpy.average(visslice.time)
            integration_time = numpy.average(visslice.integration_time)
            gaintable_rows = abs(gt.time - vistime) < integration_time / 2.0
            
            # Find the gain for this set of visibilities
            gain = gt.data['gain'][gaintable_rows]
            original = visslice.data['vis']
            applied = copy.deepcopy(original)
            for a1 in range(visslice.nants-1):
                for a2 in range(a1+1, visslice.nants):
                    applied[:, a2, a1, :, :] = gain[0, a1,:,:] * numpy.conjugate(gain[0, a2,:,:]) * \
                                               original[:, a2, a1, :, :]
                                                         
            vis.data['vis'][rows] = applied
    return vis