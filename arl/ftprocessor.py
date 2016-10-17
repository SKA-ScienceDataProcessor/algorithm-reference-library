# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing. These are built on top of the core
functions in arl.fourier_transforms
"""

import numpy

from arl.parameters import get_parameter
from arl.image_iterators import *
from arl.visibility_iterators import *
from arl.synthesis_support import invert2d, predict2d

import logging

log = logging.getLogger("arl.ftprocessor")
    
def predict_2d(vis, model, predict_function=None, params={}):
    """ Predict using image partitions, calling specified predict function

    """
    log.debug("ftprocessor.predict_2d: predicting")
    return vis


def predict_image_partition(vis, model, predict_function=predict_2d, params={}):
    """ Predict using image partitions, calling specified predict function

    """
    log.debug("ftprocessor.predict_image_partition: predicting")
    nraster = get_parameter(params, "image_partitions", 3)
    for mpatch in raster(model, nraster=nraster):
        predict_function(vis, mpatch, params=params)
    return vis


def predict_fourier_partition(vis, model, predict_function=predict_2d, params={}):
    """ Predict using fourier partitions, calling specified predict function

    """
    log.debug("ftprocessor.predict_fourier_partition: predicting")
    nraster = get_parameter(params, "fourier_partitions", 3)
    for fpatch in raster(model, nraster=nraster):
        predict_function(vis, fpatch, params=params)
    return vis


def predict_wslice_partition(vis, model, predict_function=predict_2d, params={}):
    """ Predict using wslices

    """
    log.debug("ftprocessor.predict_wslice_partition: predicting")
    wslice = get_parameter(params, "wslice", 1000)
    for vslice in vis_wslice_iter(vis, wslice):
        predict_function(vslice, model, params=params)
    
    return vis


def invert_2d(vis, dirty, psf, sumofweights, invert_function=None, params={}):
    """ Predict using image partitions, calling specified predict function

    """
    
    log.debug("ftprocessor.invert_2d: inverting")
    return dirty, psf, sumofweights


def invert_image_partition(vis, dirty, psf, sumofweights, invert_function=invert_2d, params={}):
    """ Predict using image partitions, calling specified predict function

    """
    
    log.debug("ftprocessor.invert_image_partition: inverting")
    nraster = get_parameter(params, "image_partitions", 3)
    for dpatch in raster(dirty, nraster=nraster):
        invert_function(vis, dpatch, psf, sumofweights, params=params)
    
    return dirty, psf, sumofweights


def invert_fourier_partition(vis, dirty, psf, sumofweights, invert_function=invert_2d, params={}):
    """ Predict using fourier partitions, calling specified predict function

    """
    log.debug("ftprocessor.invert_fourier_partition: inverting")
    nraster = get_parameter(params, "fourier_partitions", 3)
    for dpatch in raster(dirty, nraster=nraster):
        invert_function(vis, dirty, psf, sumofweights, params=params)
    
    return dirty, psf, sumofweights

def invert_wslice_partition(vis, dirty, psf, sumofweights, invert_function=invert_2d, params={}):
    """ Predict using wslices

    """
    log.debug("ftprocessor.invert_wslice_partition: inverting")
    wstep = get_parameter(params, "wstep", 1000)
    for visslice in vis_wslice_iter(vis, wstep):
        invert_function(visslice, dirty, psf, sumofweights, params=params)
    
    return dirty, psf, sumofweights