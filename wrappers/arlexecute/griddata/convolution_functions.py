#
"""
Functions that define and manipulate ConvolutionFunctions.

The griddata has axes [chan, pol, z, dy, dx, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
order in the WCS is reversed so the grid_WCS describes UU, VV, DUU, DVV, WW, STOKES, FREQ axes.

GridData can be used to hold the Fourier transform of an Image or gridded visibilities. In addition, the convolution
function can be stored in a GridData, most probably with finer spatial sampling.


"""

from processing_components.griddata import create_convolutionfunction_from_array, \
    create_convolutionfunction_from_image, convert_convolutionfunction_to_image, \
    apply_bounding_box_convolutionfunction, calculate_bounding_box_convolutionfunction, \
    qa_convolutionfunction