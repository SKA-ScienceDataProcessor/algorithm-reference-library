# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#

import numpy

from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord

from arl.image_operations import import_image_from_fits
from arl.visibility_operations import combine_visibility
from arl.fourier_transforms import predict_visibility, invert_visibility
from arl.data_models import *
from arl.parameters import get_parameter

def create_skycomponent(direction: SkyCoord, flux: numpy.array, frequency: numpy.array, shape: str = 'Point',
                        param: dict = None, name: str = ''):
    """ A single SkyComponent with direction, flux, shape, and params for the shape

    :param direction:
    :type SkyCoord:
    :param flux:
    :type numpy.array:
    :param frequency:
    :type numpy.array:
    :param shape: 'Point' or 'Gaussian'
    :type str:
    :param name:
    :type str:
    :returns: SkyComponent
    """
    sc = SkyComponent()
    sc.direction = direction
    sc.frequency = frequency
    sc.name = name
    sc.flux = numpy.array(flux)
    sc.shape = shape
    sc.params = param
    sc.name = name
    return sc


def find_skycomponent(im: Image, params={}):
    """ Find components in Image, return SkyComponent, just find the peak for now

    :param im: Image to be searched
    :type Image:
    :returns: SkyComponent
    """
    # TODO: Implement full image fitting of components
    print("imaging.point_source_find: Finding components in Image")
    
    # Beware: The index sequencing is opposite in wcs and Python!
    locpeak = numpy.array(numpy.unravel_index((numpy.abs(im.data)).argmax(), im.data.shape))
    print("imaging.point_source_find: Found peak at pixel coordinates %s" % str(locpeak))
    w = im.wcs.sub(['longitude', 'latitude'])
    sc = pixel_to_skycoord(locpeak[3], locpeak[2], im.wcs, 0, 'wcs')
    print("imaging.point_source_find: Found peak at world coordinates %s" % str(sc))
    flux = im.data[:, :, locpeak[2], locpeak[3]]
    print("imaging.point_source_find: Flux is %s" % flux)
    # We also need the frequency values
    w = im.wcs.sub(['spectral'])
    frequency = w.wcs_pix2world(range(im.data.shape[0]), 1)
    return create_skycomponent(direction=sc, flux=flux, frequency=frequency, shape='point')


def fit_skycomponent(im: Image, sc: SkyCoord, params={}):
    """ Find flux at a given direction, return SkyComponent

    :param im:
    :type Image:
    :param sc:
    :type SkyCoord:
    :returns: SkyComponent

    """
    print("imaging.find_flux_at_direction: Extracting flux at world coordinates %s" % str(sc))
    w = im.wcs.sub(['longitude', 'latitude'])
    pixloc = skycoord_to_pixel(sc, im.wcs, 0, 'wcs')
    print("imaging.find_flux_at_direction: Extracting flux at pixel coordinates %d %d" % (pixloc[0], pixloc[1]))
    flux = im.data[:, :, int(pixloc[1] + 0.5), int(pixloc[0] + 0.5)]
    print("imaging.find_flux_at_direction: Flux is %s" % flux)
    
    # We also need the frequency values
    w = im.wcs.sub(['spectral'])
    frequency = w.wcs_pix2world(range(im.data.shape[0]), 0)
    
    return create_skycomponent(direction=sc, flux=flux, frequency=frequency, shape='point')


def add_skymodels(sm1: SkyModel, sm2: SkyModel):
    """ Add two sky models together
    
    :param sm1:
    :type SkyModel:
    :param sm2:
    :type SkyModel:
    :returns: SkyModel
    """
    fsm = SkyModel()
    fsm.images = [sm1.images, sm2.images]
    fsm.components = [sm1.components, sm2.components]
    return fsm


def create_skymodel_from_image(im: Image):
    """ Create a skymodel from an image or image
    
    :param im:
    :type Image:
    :returns: SkyModel
    """
    sm = SkyModel()
    sm.images.append(im)
    return sm


def add_image_to_skymodel(sm: SkyModel, im: Image):
    """Add images to a sky model
    
    :param sm:
    :type SkyModel:
    :param im:
    :type Image:
    :returns: SkyModel
    """
    sm.images.append(im)
    return sm


def create_skymodel_from_component(comp: SkyComponent):
    """Create sky model from component
    
    :param comp:
    :type SkyComponent:
    :returns: SkyModel
    """
    sm = SkyModel()
    sm.components.append(comp)
    return sm


def add_component_to_skymodel(sm: SkyModel, comp: SkyComponent):
    """Add Component to a sky model
    
    :param sm:
    :type SkyModel:
    :param comp:
    :type SkyComponent:
    :returns: SkyModel
   """
    sm.components.append(comp)
    return sm


def solve_skymodel(vis: Visibility, sm: SkyModel, deconvolver, params={}):
    """Solve for SkyModel using a deconvolver. The interface of deconvolver is the same as clean.

    This is the same as a majorcycle.

    :param vis:
    :type Visibility: Visibility to be processed
    :param sm:
    :type SkyModel:
    :param deconvolver: Deconvolver to be used e.g. msclean
    :arg function:
    :returns: Visibility, SkyModel
    """
    nmajor = get_parameter(params, 'nmajor', 5)
    print("solve_combinations.solve_skymodel: Performing %d major cycles" % nmajor)
    
    # The model is added to each major cycle and then the visibilities are
    # calculated from the full model
    vispred = predict_visibility(vis, sm, params={})
    visres = combine_visibility(vis, vispred, 1.0, -1.0)
    dirty, psf, sumwt = invert_visibility(visres, params={})
    thresh = get_parameter(params, "threshold", 0.0)
    
    comp = sm.images[0]
    for i in range(nmajor):
        print("solve_combinations.solve_skymodel: Start of major cycle %d" % i)
        cc, res = deconvolver(dirty, psf, params={})
        comp += cc
        vispred = predict_visibility(vis, sm, params={})
        visres = combine_visibility(vis, vispred, 1.0, -1.0)
        dirty, psf, sumwt = invert_visibility(visres, params={})
        if numpy.abs(dirty.data).max() < 1.1 * thresh:
            print("Reached stopping threshold %.6f Jy" % thresh)
            break
        print("solve_combinations.solve_skymodel: End of major cycle")
    print("solve_combinations.solve_skymodel: End of major cycles")
    return visres, sm

def solve_skymodel_gains(vis: Visibility, sm: SkyModel, deconvolver, params={}):
    """Solve for SkyModel a deconvolver. The interface of deconvolver is the same as clean.

    This is the same as self-calibration

    :param vis:
    :type Visibility: Visibility to be processed
    :param sm:
    :type SkyModel:
    :param deconvolver: Deconvolver to be used e.g. msclean
    :arg function:
    :returns: Visibility, SkyModel, Gaintable
    """
    print("solve_combinations.solve_skymodel_gains: not implemeneted yet")
    return vis, sm, GainTable()


if __name__ == '__main__':
    import os

    os.chdir('../')
    print(os.getcwd())

    flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.5, 0.0, 0.0, 0.0]])
    frequency = numpy.array([1.0e8, 1.5e8])
    direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
    comp = create_skycomponent(direction, flux, frequency, shape='Point', name="Mysource")

    kwargs = {}
    m31image = import_image_from_fits("./data/models/M31.MOD")
    m31im = SkyModel()
    m31im.images.append(m31image)
    flux = numpy.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    direction = SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')
    comp = create_skycomponent(direction, flux, frequency=numpy.arange(5e6, 300e6, 1e7), shape='Point',
                               name="Mysource")
    m31comp = SkyModel()
    m31comp.components.append(comp)
    m31added = add_skymodels(m31im, m31comp)
