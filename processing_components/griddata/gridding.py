""" Imaging is based on used of the FFT to perform Fourier transforms efficiently. Since the observed visibility data_models
do not arrive naturally on grid points, the sampled points are resampled on the FFT grid using a convolution function to
smear out the sample points. The resulting grid points are then FFT'ed. The result can be corrected for the gridding
convolution function by division in the image plane of the transform.

This approach may be extended to include image plane effect such as the w term and the antenna/station primary beam.

This module contains functions for performing the gridding process and the inverse degridding process.
"""

import logging

import numpy

log = logging.getLogger(__name__)


def grid_visibility_to_griddata(vis, griddata, cf, **kwargs):
    """Grid Visibility onto a GridData

    :param vis: Visibility to be gridded
    :param griddata: GridData
    :param cf: Convolution function (as GridData)
    :param kwargs:
    :return:
    """
    return griddata


def degrid_visibility_from_griddata(vis, griddata, **kwargs):
    """Degrid Visibility from a GridData

    :param vis: Visibility to be degridded
    :param griddata: GridData containing image
    :param cf: Convolution function (as GridData)
    :param kwargs:
    :return:
    """
    return vis
