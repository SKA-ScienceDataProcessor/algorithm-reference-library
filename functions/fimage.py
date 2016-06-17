# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
import numpy as numpy

from astropy.wcs import WCS
from astropy.nddata import NDData
from astropy.io import fits

from functions.fconfiguration import fcontext

class fimage():
    """Image class with image data and coordinates
    """
    def from_array(self, image: NDData, wcs: WCS=None, context: fcontext = None, **kwargs):
        assert len(image.shape) < 5, "Too many axes in image"
        self.data=image
        self.wcs=WCS

    def from_fits(self, fitsfile: str):
        hdulist=fits.open(fitsfile)
        self.data=hdulist[0].data
        self.wcs=WCS(fitsfile)

if __name__ == '__main__':
    import os
    print(os.getcwd())
    m31model = fimage()
    m31model.from_fits("../data/models/m31.model.fits")
    print(m31model.data.shape)
    print(m31model.wcs)