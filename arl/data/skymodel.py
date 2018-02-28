import logging

from arl.calibration.operations import copy_gaintable
from arl.image.operations import copy_image
from arl.skycomponent.base import copy_skycomponent

log = logging.getLogger(__name__)


class SkyModel:
    """ A model for the sky including gain information
    """
    
    def __init__(self, images=None, components=None, gt=None):
        """ Holds a model of the sky including gain information
        
        """
        if images is not None:
            self.images = [(copy_image(im), copy_gaintable(gt)) for im in images]
        else:
            self.images = None
        
        if components is not None:
            self.components = [(copy_skycomponent(sc), copy_gaintable(gt)) for sc in components]
        else:
            self.components = None
    
    def append_component(self, component, gt):
        """ Append component to sky model

        """
        if self.components is None:
            self.components = list()
        
        self.components.append((component, gt))
    
    def append_image(self, im, gt):
        """ Append component to sky model

        """
        if self.images is None:
            self.images = list()
        
        self.images.append((im, gt))
    
    def __str__(self):
        """Default printer for SkyModel

        """
        s = "SkyModel:\n"
        if self.components is not None:
            for i, sc in enumerate(self.components):
                s += str(sc[0])
                s += str(sc[1])
            s += "\n"
        
        if self.images is not None:
            for i, im in enumerate(self.images):
                s += str(im[0])
                s += str(im[1])
            s += "\n"
        
        return s
