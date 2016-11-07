# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord

from arl.data.data_models import *
from arl.data.parameters import *

log = logging.getLogger("arl.skymodel_operations")

def create_skycomponent(direction: SkyCoord, flux: numpy.array, frequency: numpy.array, shape: str = 'Point',
                        param: dict = None, name: str = ''):
    """ A single Skycomponent with direction, flux, shape, and params for the shape

    :param param:
    :param direction:
    :param flux:
    :param frequency:
    :param shape: 'Point' or 'Gaussian'
    :param name:
    :returns: Skycomponent
    """
    sc = Skycomponent()
    sc.direction = direction
    sc.frequency = frequency
    sc.name = name
    sc.flux = numpy.array(flux)
    sc.shape = shape
    sc.params = param
    sc.name = name
    return sc


def find_skycomponent(im: Image, params=None):
    """ Find components in Image, return Skycomponent, just find the peak for now

    :param params:
    :param im: Image to be searched
    :returns: Skycomponent
    """
    # TODO: Implement full image fitting of components
    if params is None:
        params = {}
    log_parameters(params)
    log.debug("point_source_find: Finding components in Image")
    
    # Beware: The index sequencing is opposite in wcs and Python!
    locpeak = numpy.array(numpy.unravel_index((numpy.abs(im.data)).argmax(), im.data.shape))
    log.debug("point_source_find: Found peak at pixel coordinates %s" % str(locpeak))
    sc = pixel_to_skycoord(locpeak[3], locpeak[2], im.wcs, 0, 'wcs')
    log.debug("point_source_find: Found peak at world coordinates %s" % str(sc))
    flux = im.data[:, :, locpeak[2], locpeak[3]]
    log.debug("point_source_find: Flux is %s" % flux)
    # We also need the frequency values
    w = im.wcs.sub(['spectral'])
    frequency = w.wcs_pix2world(range(im.data.shape[0]), 1)
    return create_skycomponent(direction=sc, flux=flux, frequency=frequency, shape='point')


def fit_skycomponent(im: Image, sc: SkyCoord, params=None):
    """ Find flux at a given direction, return Skycomponent

    :param params:
    :param im:
    :param sc:
    :returns: Skycomponent

    """
    if params is None:
        params = {}
    log_parameters(params)
    log.debug("find_flux_at_direction: Extracting flux at world coordinates %s" % str(sc))
    pixloc = skycoord_to_pixel(sc, im.wcs, 0, 'wcs')
    log.debug("find_flux_at_direction: Extracting flux at pixel coordinates %d %d" % (pixloc[0], pixloc[1]))
    flux = im.data[:, :, int(pixloc[1] + 0.5), int(pixloc[0] + 0.5)]
    log.debug("find_flux_at_direction: Flux is %s" % flux)
    
    # We also need the frequency values
    w = im.wcs.sub(['spectral'])
    frequency = w.wcs_pix2world(range(im.data.shape[0]), 0)
    
    return create_skycomponent(direction=sc, flux=flux, frequency=frequency, shape='point')


def insert_skycomponent(im: Image, sc: Skycomponent, params=None):
    """ Insert a Skycompoenet into an image

    :param params:
    :param im:
    :param sc:
    :returns: image

    """
    if params is None:
        params = {}
    assert sc.shape == 'Point', "Cannot handle shape %s"% sc.shape
    log.debug("insert_skycomponent: Inserting flux at world coordinates %s" % str(sc))
    pixloc = skycoord_to_pixel(sc.direction, im.wcs, 0, 'wcs')
    log.debug("insert_skycomponent: Inserting flux at pixel coordinates %d %d" % (pixloc[0], pixloc[1]))
    insert_method = get_parameter(params, "insert_method", "Lanczos")
    if insert_method == "Lanczos":
        _L2D(im.data, pixloc[1], pixloc[0], sc.flux)
    else:
        im.data[:, :, int(pixloc[1] + 0.5), int(pixloc[0] + 0.5)] += sc.flux
       
    return im


def add_skymodels(sm1: Skymodel, sm2: Skymodel):
    """ Add two sky models together
    
    :param sm1:
    :param sm2:
    :returns: Skymodel
    """
    fsm = Skymodel()
    fsm.images = [sm1.images, sm2.images]
    fsm.components = [sm1.components, sm2.components]
    return fsm


def create_skymodel_from_image(im: Image):
    """ Create a skymodel from an image or image
    
    :param im:
    :returns: Skymodel
    """
    sm = Skymodel()
    sm.images.append(im)
    return sm


def add_image_to_skymodel(sm: Skymodel, im: Image):
    """Add images to a sky model
    
    :param sm:
    :param im:
    :returns: Skymodel
    """
    sm.images.append(im)
    return sm


def create_skymodel_from_component(comp: Skycomponent):
    """Create sky model from component
    
    :param comp:
    :returns: Skymodel
    """
    sm = Skymodel()
    sm.components.append(comp)
    return sm


def add_component_to_skymodel(sm: Skymodel, comp: Skycomponent):
    """Add Component to a sky model
    
    :param sm:
    :param comp:
    :returns: Skymodel
   """
    sm.components.append(comp)
    return sm


def _L2D(im, x, y, flux, a = 7):
    """Perform Lanczos interpolation onto a grid
    
    """
    
    nchan, npol, ny, nx = im.shape
    a=int(a)
    intx = int(numpy.floor(x))
    inty = int(numpy.floor(y))
    fracx = x - intx
    fracy = y - inty
    gridx = numpy.arange(-a, a)
    gridy = numpy.arange(-a, a)

    insert = numpy.zeros([2 * a + 1, 2 * a + 1])
    for iy in gridy:
        insert[iy, gridx + a] = _L(gridx + fracx) * _L(iy + fracy)
    insertsum = numpy.sum(insert)
    assert insertsum > 0, "Sum of interpolation coefficients %g" % insertsum
    insert = insert / insertsum

    for chan in range(nchan):
        for pol in range(npol):
            for iy in gridy:
                im[chan, pol, iy + inty, gridx + intx] += flux[chan,pol] * insert[iy,gridx+a]
            
    return im
    

def _sinc(x):
    s = numpy.zeros_like(x)
    s[x != 0.0] = numpy.sin(numpy.pi*x[x != 0.0])/(numpy.pi*x[x != 0.0])
    return s


def _L(x, a = 5):
    L = _sinc(x) *_sinc(x/a)
    return L