#
# MeasurementSets V2 Reference Codes Based on Python-casacore
#

import os
import gc
import re
import glob
import math
import numpy
import shutil
import scipy
from scipy.constants import speed_of_light
from datetime import datetime
from collections import OrderedDict

from data_models.memory_data_models import Visibility, BlockVisibility, Configuration

__version__ = '0.1'
__revision__ = '$Rev$'
__all__ = ['STOKES_CODES', 'NUMERIC_STOKES']

STOKES_CODES = {'I': 1, 'Q': 2, 'U': 3, 'V': 4,
                'RR': 5, 'RL': 6, 'LR': 7, 'LL': 8,
                'XX': 9, 'XY': 10, 'YX': 11, 'YY': 12}

NUMERIC_STOKES = {1: 'I', 2: 'Q', 3: 'U', 4: 'V',
                  5: 'RR', 6: 'RL', 7: 'LR', 8: 'LL',
                  9: 'XX', 10: 'XY', 11: 'YX', 12: 'YY'}


def geo_to_ecef(lat, lon, elev):
    """
    Convert latitude (rad), longitude (rad), elevation (m) to earth-
    centered, earth-fixed coordinates.
    """

    WGS84_a = 6378137.00000000
    WGS84_b = 6356752.31424518
    N = WGS84_a ** 2 / numpy.sqrt(WGS84_a ** 2 * numpy.cos(lat) ** 2 + WGS84_b ** 2 * numpy.sin(lat) ** 2)
    x = (N + elev) * numpy.cos(lat) * numpy.cos(lon)
    y = (N + elev) * numpy.cos(lat) * numpy.sin(lon)
    z = ((WGS84_b ** 2 / WGS84_a ** 2) * N + elev) * numpy.sin(lat)

    return (x, y, z)


def get_eci_transform(lat):
    """
    Return a 3x3 transformation matrix that converts a baseline in
    [east, north, elevation] to earth-centered inertial coordinates
    for that baseline [x, y, z].
    """
    return numpy.array([[0.0, -numpy.sin(lat), numpy.cos(lat)],
                        [1.0, 0.0, 0.0],
                        [0.0, numpy.cos(lat), numpy.sin(lat)]])


def _cmp_to_lt(self, other):
    """
    Return a < b.  Compute by @cmp_to_total from __cmp__
    """

    return True if self.__cmp__(other) < 0 else False


def _cmp_to_le(self, other):
    """
    Return a <= b.  Compute by @cmp_to_total from __cmp__
    """

    return True if self.__cmp__(other) <= 0 else False


def _cmp_to_gt(self, other):
    """
    Return a > b.  Compute by @cmp_to_total from __cmp__
    """

    return True if self.__cmp__(other) > 0 else False


def _cmp_to_ge(self, other):
    """
    Return a >= b.  Compute by @cmp_to_total from __cmp__
    """

    return True if self.__cmp__(other) >= 0 else False


def _cmp_to_eq(self, other):
    """
    Return a == b.  Compute by @cmp_to_total from __cmp__
    """

    return True if self.__cmp__(other) == 0 else False


def _cmp_to_ne(self, other):
    """
    Return a != b.  Compute by @cmp_to_total from __cmp__
    """

    return True if self.__cmp__(other) != 0 else False


def cmp_to_total(cls):
    """
    Decorator to define the six comparison operators that are needed
    for total ordering that can be used in all versions of Python
    from the __cmp__() method.
    """

    names = ['__lt__', '__le__', '__gt__', '__ge__', '__eq__', '__ne__']
    funcs = [_cmp_to_lt, _cmp_to_le, _cmp_to_gt, _cmp_to_ge, _cmp_to_eq, _cmp_to_ne]

    for name, func in zip(names, funcs):
        # Is it defined?
        if name not in dir(cls):
            func.__name__ = name
            setattr(cls, name, func)

    return cls


def merge_baseline(ant1, ant2, shift=16):
    """
    Merge two stand ID numbers into a single baseline using the specified bit
    shift size.
    """
    return (ant1 << shift) | ant2

def split_baseline(baseline, shift=16):
    """
    Given a baseline, split it into it consistent stand ID numbers.
    """

    part = 2 ** shift - 1
    return (baseline >> shift) & part, baseline & part