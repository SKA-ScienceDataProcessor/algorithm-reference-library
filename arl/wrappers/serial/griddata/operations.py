#
"""
Functions that define and manipulate GridData.

The griddata has axes [chan, pol, z, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes.

GridData can be used to hold the Fourier transform of an Image or gridded visibilities. In addition, the convolution
function can be stored in a GridData, most probably with finer spatial sampling.


"""

from arl.processing_components.griddata.operations import griddata_sizeof, create_griddata_from_array, \
    create_griddata_from_image, convert_griddata_to_image, qa_griddata, copy_griddata