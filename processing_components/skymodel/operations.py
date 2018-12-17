"""Function to manage skymodels.

"""

import logging

import numpy

import matplotlib.pyplot as plt

from astropy.wcs.utils import skycoord_to_pixel

from data_models.memory_data_models import SkyModel, GainTable
from processing_library.image.operations import copy_image
from ..image.operations import show_image, smooth_image
from ..skycomponent.base import copy_skycomponent
from ..calibration.operations import copy_gaintable
from ..skycomponent.operations import filter_skycomponents_by_flux, insert_skycomponent, image_voronoi_iter

log = logging.getLogger(__name__)


def copy_skymodel(sm):
    """ Copy a sky model
    
    """
    return SkyModel(components=[copy_skycomponent(comp) for comp in sm.components],
                    image=copy_image(sm.image),
                    gaintable=copy_gaintable(sm.gaintable),
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
                    image=copy_image(im),
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


def partition_skymodel_by_voronoi(sc, model, flux_threshold=-numpy.inf):
    """Partition skymodel according to flux

    :param sc:
    :param model:
    :param flux_threshold:
    :return:
    """
    brightsc = filter_skycomponents_by_flux(sc, flux_min=flux_threshold)
    log.info('Partitioning uses %d components to define a Voronoi tesselation'
             % (len(brightsc)))
    return SkyModel(images=[copy_image(im) for im in image_voronoi_iter(model, brightsc)],
                    fixed=False)
