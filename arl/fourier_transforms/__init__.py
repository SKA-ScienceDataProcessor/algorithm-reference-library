"""FFT support functions

All grids and images are considered quadratic and centered around
`npixel//2`, where `npixel` is the pixel width/height. This means that `npixel//2` is
the zero frequency for FFT purposes, as is convention. Note that this
means that for even `npixel` the grid is not symmetrical, which means that
e.g. for convolution kernels odd image sizes are preferred.

This is implemented for reference in
`coordinates`/`coordinates2`. Some noteworthy properties:
- `ceil(field_of_view * lam)` gives the image size `npixel` in pixels
- `lam * coordinates2(npixel)` yields the `u,v` grid coordinate system
- `field_of_view * coordinates2(npixel)` yields the `l,m` image coordinate system
   (radians, roughly)
"""
