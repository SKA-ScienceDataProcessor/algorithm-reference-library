# Tim Cornwell <realtimcornwell@gmail.com>
#
# Visibility data structure: a Table with columns ['uvw', 'time', 'antenna1', 'antenna2', 'vis', 'weight']
# and an attached attribute which is the frequency of each channel

from astropy import constants as const
from astropy.coordinates import SkyCoord, CartesianRepresentation
from astropy.table import Table, vstack

from crocodile.simulate import *
from arl.simulate_visibility import Configuration, create_named_configuration

"""
Functions that represent a visibility set.

The data structure:
- an AstroPy Table with columns ['uvw', 'time', 'antenna1', 'antenna2', 'vis', 'weight']
- An attached attribute which is the frequency of each channel as a numy array
- An attached attribute which is the phase centre as an AstroPy SkyCoord
"""


class AQ:
    """
    Gain table with time, antenna, gain[:,chan,pol] columns
    """
    
    # TODO: Implement gaintables with Jones and Mueller matrices
    
    def __init__(self):
        self.data = None
        self.context = NONE


def aq_visibility(vt, **kwargs):
    """Assess the quality of an image

    :param vt:
    :type Visibility:
    :returns: AQ
    """
    print("assess_quality.aq_visibility: not yet implemented")
    return AQ()


def aq_image(im, **kwargs):
    """Assess the quality of an image

    :param im:
    :type Image:
    :returns: AQ
    """
    print("assess_quality.aq_image: not yet implemented")
    return AQ()


def aq_gaintable(gt, **kwargs):
    """Assess the quality of a gaintable

    :param im:
    :type GainTable:
    :returns: AQ
    """
    print("assess_quality.aq_gaintable: not yet implemented")
    return AQ()

