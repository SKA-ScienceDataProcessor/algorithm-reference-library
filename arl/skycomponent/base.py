"""Function to manage skycomponents.

"""

import collections
import logging

from arl.data.data_models import Skycomponent

log = logging.getLogger(__name__)


def copy_skycomponent(sc):
    """Copy a sky component of Iterable of skycomponents
    
    :param sc:
    :return:
    """
    single = not isinstance(sc, collections.Iterable)
    
    if single:
        return Skycomponent(
            direction=sc.direction,
            frequency=sc.frequency,
            name=sc.name,
            flux=sc.flux,
            shape=sc.shape,
            params=sc.params,
            polarisation_frame=sc.polarisation_frame)
    else:
        return [Skycomponent(
            direction=s.direction,
            frequency=s.frequency,
            name=s.name,
            flux=s.flux,
            shape=s.shape,
            params=s.params,
            polarisation_frame=s.polarisation_frame) for s in sc]
