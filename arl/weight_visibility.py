# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
import numpy as numpy

from astropy.table import Table

import os

import numpy

import astropy.units as units
from astropy.coordinates import EarthLocation
from astropy.table import Table, Column, vstack

from crocodile.simulate import *

"""
Functions that weight the visibility data using various algorithms
"""


def briggs_weight(vt, im, **kwargs):
    """ Reweight the visibility data in place using Briggs' algorithm

    :param vt:
    :type Visibility:
    :param im:
    :type Image:
    :param kwargs:
    :returns: Configuration
    """
    print("weight_visibility.briggs_weight: not yet implemented")
    return vt


def uniform_weight(vt, im, **kwargs):
    """ Reweight the visibility data in place using uniform weighting

    :param vt:
    :type Visibility:
    :param im:
    :type Image:
    :param kwargs:
    :returns: Configuration
    """
    print("weight_visibility.briggs_weight: not yet implemented")
    return vt