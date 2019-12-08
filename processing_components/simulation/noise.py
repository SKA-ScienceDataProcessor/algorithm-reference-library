"""
Functions that add noise

"""

__all__ = ['calculate_noise_blockvisibility', 'calculate_noise_visibility', 'addnoise_visibility']

import csv
import logging
from typing import List

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from scipy import interpolate

from data_models.memory_data_models import Configuration, Image, GainTable, Skycomponent, SkyModel, PointingTable
from data_models.memory_data_models import Visibility, BlockVisibility
from data_models.parameters import arl_path
from data_models.polarisation import PolarisationFrame
from processing_components.calibration.calibration_control import create_calibration_controls
from processing_components.calibration.operations import create_gaintable_from_blockvisibility, apply_gaintable
from processing_components.image.operations import import_image_from_fits
from processing_components.imaging.base import predict_2d, predict_skycomponent_visibility, \
    create_image_from_visibility, advise_wide_field
from processing_components.imaging.primary_beams import create_pb
from processing_components.skycomponent.operations import create_skycomponent, insert_skycomponent, \
    apply_beam_to_skycomponent, filter_skycomponents_by_flux
from processing_components.visibility.base import create_blockvisibility, create_visibility
from processing_components.visibility.coalesce import convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility
from processing_library.image.operations import create_image_from_array

log = logging.getLogger(__name__)

def calculate_noise_visibility(bandwidth, int_time, diameter, t_sys, eta):
    """Determine noise rms per visibility
    :returns: Sigma [nrows]
    """
    
    k_b = 1.38064852e-23
    area = numpy.pi * (diameter / 2.) ** 2
    bt = bandwidth * int_time
    sigma = (numpy.sqrt(2) * k_b * t_sys) / (area * eta * (numpy.sqrt(bt)))
    sigma *= 1e26
    return sigma


def calculate_noise_blockvisibility(bandwidth, int_time, diameter, t_sys, eta):
    """Determine noise rms per visibility
    :returns: Sigma [nrows, nchan]
    """
    
    k_b = 1.38064852e-23
    area = numpy.pi * (diameter / 2.) ** 2
    bt = numpy.outer(int_time, bandwidth)
    sigma = (numpy.sqrt(2) * k_b * t_sys) / (area * eta * (numpy.sqrt(bt)))
    sigma *= 1e26
    return sigma


def addnoise_visibility(vis, t_sys=None, eta=None):
    """ Add noise to a visibility
    
    TODO: Obtain sensitivity values from vis as a function of frequency
    
    :param vis:
    :param t_sys: System temperature
    :param eta: Efficiency
    :return:
    """
    assert isinstance(vis, Visibility) or isinstance(vis, BlockVisibility), vis
    
    if t_sys is None:
        t_sys = 20.0
    
    if eta is None:
        eta = 0.78
    
    # We need to handle Visibility and BlockVisibility separately since time and bandwidth are
    # stored differently
    if isinstance(vis, Visibility):
        sigma = calculate_noise_visibility(vis.data['channel_bandwidth'], vis.data['integration_time'],
                                           vis.configuration.diameter[0], t_sys=t_sys, eta=eta)
        log.debug('addnoise_visibility: RMS noise value: %g' % sigma[0])
        # Each pol gets a separate noise
        for pol in range(vis.npol):
            vis.data["vis"][:, pol].real += numpy.random.normal(0, sigma)
            vis.data["vis"][:, pol].imag += numpy.random.normal(0, sigma)
    elif isinstance(vis, BlockVisibility):
        sigma = calculate_noise_blockvisibility(vis.channel_bandwidth, vis.data['integration_time'],
                                                vis.configuration.diameter[0], t_sys=t_sys, eta=eta)
        log.debug('addnoise_visibility: RMS noise value (first integration, first channel): %g' % sigma[0, 0])
        for row in range(vis.nvis):
            for ant1 in range(vis.nants):
                for ant2 in range(ant1, vis.nants):
                    for pol in range(vis.npol):
                        vis.data["vis"][row, ant2, ant1, :, pol].real += numpy.random.normal(0, sigma[row, ...])
                        vis.data["vis"][row, ant2, ant1, :, pol].imag += numpy.random.normal(0, sigma[row, ...])
                        vis.data["vis"][row, ant1, ant2, :, pol] = \
                            numpy.conjugate(vis.data["vis"][row, ant2, ant1, :, pol])
    
    return vis
