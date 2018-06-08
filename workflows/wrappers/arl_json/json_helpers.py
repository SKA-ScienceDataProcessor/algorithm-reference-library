""" Helper functions for converting JSON to ARL objects"""

from astropy.coordinates import SkyCoord
from astropy.units import Quantity, Unit
import numpy

def json_to_skycoord(d):
    """Convert JSON string to SkyCoord
    
    e.g. "ra": {"value": 30.0, "unit":"deg"},


    :param d:
    :return:
    """
    return SkyCoord(ra=json_to_quantity(d["ra"]),
                    dec=json_to_quantity(d["dec"]),
                    equinox=d["equinox"],
                    frame=d["frame"])


def json_to_quantity(q):
    """Convert JSON string to Quantity

    :param q:
    :return:
    """
    return Quantity(q["value"], Unit(q["unit"]))

def json_to_linspace(l):
    """Convert JSON string to numpy.linspace
    
    :param l:
    :return:
    """
    return numpy.linspace(l["start"], l["stop"], l["steps"])