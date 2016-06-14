# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
import numpy as numpy

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.table import Table, Row, Column, MaskedColumn, TableColumns, TableFormatter
from astropy.nddata import NDData

class fcomponent():
    """ Component with SkyCoord, NDData
    """
    def __init__(self, direction: SkyCoord, flux: NDData, shape: str = 'Point', params : dict={}):
        """
        A single component with direcction, flux, shape, and parameters for the shape
        :param dir:
        :param flux:
        :param shape:
        :param params:
        """
        self.direction=direction
        self.flux=flux
        self.shape=shape
        self.params=params


if __name__ == '__main__':
    import os
    print(os.getcwd())
    flux=NDData(data=[[1.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0]])
    direction=SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
    comp=fcomponent(direction, flux, shape='Point')
    print (comp.direction, comp.flux, comp.shape)
    print(Table(data=Column([flux]),names=Column(['flux'])))
