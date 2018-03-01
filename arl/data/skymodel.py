import logging

from arl.image.operations import copy_image
from arl.skycomponent.base import copy_skycomponent

log = logging.getLogger(__name__)

class SkyModel:
    """ A model for the sky
    """
    
    def __init__(self, images=None, components=None):
        """ Holds a model of the sky
        
        """
        if images is not None:
            self.images = [copy_image(im) for im in images]
        else:
            self.images = None
        
        if components is not None:
            self.components = [copy_skycomponent(sc) for sc in components]
        else:
            self.components = None
            
    def __str__(self):
        """Default printer for SkyModel

        """
        s = "SkyModel:\n"
        if self.components is not None:
            for i, sc in enumerate(self.components):
                s += str(sc)
            s += "\n"
        
        if self.images is not None:
            for i, im in enumerate(self.images):
                s += str(im)
            s += "\n"
        
        return s