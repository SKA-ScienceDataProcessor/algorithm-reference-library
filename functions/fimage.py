# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
import numpy as numpy

from astropy.wcs import WCS
from astropy.nddata import NDData
from astropy.io import fits

class fimage():
    """
    Image class with image data (as a numpy.array) and optionally WCS
    """

    def __init__(self):
        """
        :type image: numpy.array
        """
        self.data = None
        self.wcs = None

    def from_array(self, image: numpy.array, wcs: WCS = None):
        """
        :type image: numpy.array
        """
        assert len(image.shape) < 5, "Too many axes in image"
        self.data = image
        self.wcs = None
        return self

    def from_fits(self, fitsfile: str):
        hdulist = fits.open(fitsfile)
        self.data = hdulist[0].data
        self.wcs = WCS(fitsfile)
        return self

    def addwcs(self, wcs: WCS):
        self.wcs = WCS

    def __add__(self, other):
        self.data+=other.data
        return self

if __name__ == '__main__':
    m31model = fimage()
    m31model.from_fits("../data/models/m31.model.fits")
    m31model_by_array = fimage()
    m31model_by_array.from_array(m31model.data, m31model.wcs)
    m31model += m31model_by_array
    print(m31model.data.shape)
    print(m31model.wcs)
