""" Imaging is based on used of the FFT to perform Fourier transforms efficiently. Since the observed visibility data_models
do not arrive naturally on grid points, the sampled points are resampled on the FFT grid using a convolution function to
smear out the sample points. The resulting grid points are then FFT'ed. The result can be corrected for the griddata
convolution function by division in the image plane of the transform.

This module contains functions for performing the griddata process and the inverse degridding process.

The GridData data model is used to hold the specification of the desired result.
"""

import logging

import numpy
import numpy.testing

from arl.processing_library.image.operations import ifft, fft, create_image_from_array
from arl.processing_components.visibility.operations import copy_visibility

log = logging.getLogger(__name__)

from arl.processing_components.griddata.gridding import convolution_mapping, grid_visibility_to_griddata, \
    grid_visibility_to_griddata_fast, grid_weight_to_griddata, degrid_visibility_from_griddata, \
    fft_griddata_to_image, fft_image_to_griddata, griddata_reweight, griddata_merge_weights
