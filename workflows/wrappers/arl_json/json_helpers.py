""" Helper functions for converting JSON to ARL objects

"""

from astropy.coordinates import SkyCoord
from astropy.units import Quantity, Unit

from processing_components.image.operations import export_image_to_fits, import_image_from_fits
from data_models.data_model_helpers import *

import numpy

def json_to_skycoord(d):
    """Convert JSON string to SkyCoord
    
    "phasecentre": {
      "ra": {
        "value": 30.0,
        "unit": "deg"
      },
      "dec": {
        "value": -60.0,
        "unit": "deg"
      },
      "frame": "icrs",
      "equinox": "j2000"
    }

    :param d:
    :return:
    """
    return SkyCoord(ra=json_to_quantity(d["ra"]),
                    dec=json_to_quantity(d["dec"]),
                    equinox=d["equinox"],
                    frame=d["frame"])


def json_to_quantity(q):
    """Convert JSON string to Quantity
    
    e.g.
    "cellsize": {
      "value": 0.001,
      "unit": "rad"
    },


    :param q:
    :return:
    """
    value = float(q["value"])
    unit = q["unit"]
    assert isinstance(unit, str), "unit must be string"
    unit = Unit(q["unit"])
    return Quantity(value, unit)

def json_to_linspace(l):
    """Convert JSON string to numpy.linspace
    
    e.g.
    "frequency": {
        "start": 0.9e8,
        "stop": 1.1e8,
        "steps": 7
    }
    
    :param l:
    :return:
    """
    nsteps = int(l["steps"])
    assert nsteps >= 0, "Number of steps cannot be less than zero %s" % str(l)
    return numpy.linspace(l["start"], l["stop"], nsteps)


def json_data_model_to_buffer(model, jbuff, dm):
    """ Send a data model to the buffer
    
    The file type is derived from the file extension. All are hdf only with the exception of Imaghe which can also be
    fits.

    :param model: Memory data model to be sent to buffer
    :param jbuff: JSON describing buffer
    :param dm: JSON describing data model
    """
    name = jbuff["directory"] + dm["name"]

    import os
    _, file_extension = os.path.splitext(dm["name"])

    if dm["data_model"] == "BlockVisibility":
        return export_blockvisibility_to_hdf5(model, name)
    elif dm["data_model"] == "Image":
        if file_extension == ".fits":
            return export_image_to_fits(model, name)
        else:
            return export_image_to_hdf5(model, name)
    elif dm["data_model"] == "SkyModel":
        return export_skymodel_to_hdf5(model, name)
    elif dm["data_model"] == "GainTable":
        return export_gaintable_to_hdf5(model, name)
    else:
        raise ValueError("Data model %s not supported" % dm["data_model"])
    
def json_buffer_to_data_model(jbuff, dm):
    """Get the data model specified in the JSON string
    
    The file type is derived from the file extension. All are hdf only with the exception of Imaghe which can also be
    fits.

    :param jbuff: JSON describing buffer
    :param dm: JSON describing data model
    :return: data model
    """
    name = jbuff["directory"] + dm["name"]

    import os
    _, file_extension = os.path.splitext(dm["name"])
    
    if dm["data_model"] == "BlockVisibility":
        return import_blockvisibility_from_hdf5(name)
    elif dm["data_model"] == "Image":
        if file_extension == ".fits":
            return import_image_from_fits(name)
        else:
            return import_image_from_hdf5(name)
    elif dm["data_model"] == "SkyModel":
        return import_skymodel_from_hdf5(name)
    elif dm["data_model"] == "GainTable":
        return import_gaintable_from_hdf5(name)
    else:
        raise ValueError("Data model %s not supported" % dm["data_model"])