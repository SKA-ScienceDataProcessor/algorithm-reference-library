"""Function to manage skymodels.

"""

__all__ = ['copy_skymodel', 'partition_skymodel_by_flux', 'show_skymodel', 'initialize_skymodel_voronoi',
           'calculate_skymodel_equivalent_image', 'update_skymodel_from_gaintables', 'update_skymodel_from_image',
           'expand_skymodel_by_skycomponents']

import logging

import matplotlib.pyplot as plt
import numpy
from astropy.wcs.utils import skycoord_to_pixel

from data_models.memory_data_models import SkyModel, GainTable
from processing_library.image.operations import copy_image
from processing_components.calibration.operations import copy_gaintable
from processing_components.image.operations import smooth_image
from processing_components.skycomponent.base import copy_skycomponent
from processing_components.skycomponent.operations import filter_skycomponents_by_flux, insert_skycomponent, image_voronoi_iter

log = logging.getLogger(__name__)


def copy_skymodel(sm):
    """ Copy a sky model
    
    """
    if sm.components is not None:
        newcomps = [copy_skycomponent(comp) for comp in sm.components]
    else:
        newcomps = None
    
    if sm.image is not None:
        newimage = copy_image(sm.image)
    else:
        newimage = None
    
    if sm.mask is not None:
        newmask = copy_image(sm.mask)
    else:
        newmask = None
    
    if sm.gaintable is not None:
        newgt = copy_gaintable(sm.gaintable)
    else:
        newgt = None
    
    return SkyModel(components=newcomps, image=newimage, gaintable=newgt, mask=newmask,
                    fixed=sm.fixed)


def partition_skymodel_by_flux(sc, model, flux_threshold=-numpy.inf):
    """Partition skymodel according to flux
    
    :param sc:
    :param model:
    :param flux_threshold:
    :return:
    """
    brightsc = filter_skycomponents_by_flux(sc, flux_min=flux_threshold)
    weaksc = filter_skycomponents_by_flux(sc, flux_max=flux_threshold)
    log.info('Converted %d components into %d bright components and one image containing %d components'
             % (len(sc), len(brightsc), len(weaksc)))
    im = copy_image(model)
    im = insert_skycomponent(im, weaksc)
    return SkyModel(components=[copy_skycomponent(comp) for comp in brightsc],
                    image=copy_image(im), mask=None,
                    fixed=False)


def show_skymodel(sms, psf_width=1.75, cm='Greys', vmax=None, vmin=None):
    sp = 1
    
    for ism, sm in enumerate(sms):
        plt.clf()
        plt.subplot(121, projection=sms[ism].image.wcs.sub([1, 2]))
        sp += 1
        
        smodel = copy_image(sms[ism].image)
        smodel = insert_skycomponent(smodel, sms[ism].components)
        smodel = smooth_image(smodel, psf_width)
        
        if vmax is None:
            vmax = numpy.max(smodel.data[0, 0, ...])
        if vmin is None:
            vmin = numpy.min(smodel.data[0, 0, ...])
        
        plt.imshow(smodel.data[0, 0, ...], origin='lower', cmap=cm, vmax=vmax, vmin=vmin)
        plt.xlabel(sms[ism].image.wcs.wcs.ctype[0])
        plt.ylabel(sms[ism].image.wcs.wcs.ctype[1])
        
        plt.title('SkyModel%d' % ism)
        
        components = sms[ism].components
        if components is not None:
            for sc in components:
                x, y = skycoord_to_pixel(sc.direction, sms[ism].image.wcs, 0, 'wcs')
                plt.plot(x, y, marker='+', color='red')
        
        gaintable = sms[ism].gaintable
        if gaintable is not None:
            plt.subplot(122)
            sp += 1
            phase = numpy.angle(sm.gaintable.gain[:, :, 0, 0, 0])
            phase -= phase[:, 0][:, numpy.newaxis]
            plt.imshow(phase, origin='lower')
            plt.xlabel('Dish/Station')
            plt.ylabel('Integration')
            plt.show()


def initialize_skymodel_voronoi(model, comps, gt=None):
    """Create a skymodel by Voronoi partitioning of the components, fill with components
    
    :param model: Model image
    :param comps: Skycomponents
    :param gt: Gaintable
    :return:
    """
    skymodel_images = list()
    for i, mask in enumerate(image_voronoi_iter(model, comps)):
        im = copy_image(model)
        im.data *= mask.data
        if gt is not None:
            newgt = copy_gaintable(gt)
            newgt.phasecentre = comps[i].direction
        else:
            newgt=None
            
        skymodel_images.append(SkyModel(image=im, components=None, gaintable=newgt, mask=mask))
    
    return skymodel_images


def calculate_skymodel_equivalent_image(sm):
    """Calculate an equivalent image for a skymodel
    
    :param sm:
    :return:
    """
    combined_model = copy_image(sm[0].image)
    combined_model.data[...] = 0.0
    for th in sm:
        if th.image is not None:
            if th.mask is not None:
                combined_model.data += th.mask.data * th.image.data
            else:
                combined_model.data += th.image.data
    
    return combined_model


def update_skymodel_from_image(sm, im, damping=0.5):
    """Update a skymodel for an image

    :param sm:
    :param im:
    :return:
    """
    for i, th in enumerate(sm):
        newim = copy_image(im)
        if th.mask is not None:
            newim.data *= th.mask.data
        th.image.data += damping * newim.data
    
    return sm


def update_skymodel_from_gaintables(sm, gt_list, calibration_context='T', damping=0.5):
    """Update a skymodel from a list of gaintables

    :param sm:
    :param im:
    :return:
    """
    assert len(sm) == len(gt_list)
    
    for i, th in enumerate(sm):
        assert isinstance(th.gaintable, GainTable), th.gaintable
        delta = numpy.exp(damping*1j*gt_list[i][calibration_context].gain)
        th.gaintable.data['gain'] *= numpy.exp(damping*1j*numpy.angle(gt_list[i][calibration_context].gain))
    
    return sm


def expand_skymodel_by_skycomponents(sm, **kwargs):
    """ Expand a sky model so that all components and the image are in separate skymodels
    
    The mask and gaintable are taken to apply for all new skymodels.
    
    :param sm: SkyModel
    :return: List of SkyModels
    """
    result = [SkyModel(components=[comp],
                       image=None,
                       gaintable=copy_gaintable(sm.gaintable),
                       mask=copy_image(sm.mask),
                       fixed=sm.fixed) for comp in sm.components]
    if sm.image is not None:
        result.append(SkyModel(components=None,
                               image=copy_image(sm.image),
                               gaintable=copy_gaintable(sm.gaintable),
                               mask=copy_image(sm.mask),
                               fixed=sm.fixed))
    return result
