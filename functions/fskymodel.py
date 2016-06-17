# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
import numpy as numpy

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.table import Table, Row, Column, MaskedColumn, TableColumns, TableFormatter
from astropy.wcs import WCS
from astropy.nddata import NDData

from functions.fcontext import fcontext
from functions.fimage import fimage
from functions.fcomponent import fcomponent

class fskymodel():
    """
    Sky components and images
    """
    def __init__(self, images: [] = None, components: [] = None) -> object:
        """

        :rtype: object
        """
        if images:
            self.images=images
        if components:
            self.components=[]

if __name__ == '__main__':
    import os
    print(os.getcwd())
    m31image = fimage().from_fits("../data/models/m31.model.fits")
    m31sm = fskymodel(m31image)
