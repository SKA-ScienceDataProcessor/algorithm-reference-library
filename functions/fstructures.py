# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
from astropy.coordinates import SkyCoord
from astropy.table import Table, Row, Column, MaskedColumn, TableColumns, TableFormatter
from astropy.wcs import WCS
from astropy.nddata import NDData

class fcoordinate(SkyCoord):
    """ Coordinates on the sky
    """

class fwcs(WCS):
    """ World Coordinate System
    """
    def __init__(self, fitsfile):
        super().__init__(fitfile)

class fvistable(Table):
    """Visibility class with uvw, time, a1, a2, vis, weight columns
    """
    def __init__(self, uvw, time, antenna1, antenna2, vis, weight, copy=False,
                 meta={'phasecentre':None}):
                 nrows = time.shape[0]
                 assert uvw.shape[0]==nrows, "Discrepancy in number of rows"
                 assert len(antenna1)==nrows, "Discrepancy in number of rows"
                 assert len(antenna2)==nrows, "Discrepancy in number of rows"
                 assert vis.shape[0]==nrows, "Discrepancy in number of rows"
                 assert weight.shape[0]==nrows, "Discrepancy in number of rows"
                 super(fvistable, self).__init__(data=[uvw, time, antenna1, antenna2, vis, weight],
                 names=['uvw', 'time', 'antenna1', 'antenna2', 'vis', 'weight'],
                 copy=copy, meta=meta)

class fimage(NDData, WCS):
    """Image class with image data and coordinates
    """
    def __init__(self, image, wcs):
        assert len(image.shape) < 5, "Too many axes in image"
        super(NDData,self).__init__(image)
        super(fwcs,self).__init__(wcs)

class fgaintable(Table):
    """Gain table with time, antenna, frequency
    """
    def __init__(self, gain, time, antenna, weight,
                 names=['gain', 'time', 'antenna', 'weight'], copy=False,
                 meta={'phasecentre':None}):
        nrows = time.shape[0]
        assert len(antenna)==nrows, "Discrepancy in number of rows"
        assert gain.shape[0]==nrows, "Discrepancy in number of rows"
        assert weight.shape[0]==nrows, "Discrepancy in number of rows"
        super.__init__(data=[gain, time, antenna, weight],
        names=['gain', 'time', 'antenna', 'weight'], copy=copy,
        meta=meta)

class fmkernel(Table):
    """ Mueller kernel with NDData, antenna1, antenna2, time
    """

class fjones(Table):
    """ Jones kernel with NDData, antenna1, antenna2, time
    """

class fcomponents(Table):
    """ Components with SkyCoord, NDData
    """

class fskymodel(Table):
    """ Sky components and images
    """
