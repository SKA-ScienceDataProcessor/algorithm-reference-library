import logging

from arl.image.operations import copy_image
from arl.skycomponent.base import copy_skycomponent

log = logging.getLogger(__name__)

class SkyModel:
    """ A model for the sky
    """
    
    def __init__(self, images=[], components=[]):
        """ A model of the sky as a list of images and a list of components
        
        """
        self.images = [copy_image(im) for im in images]
        self.components = [copy_skycomponent(sc) for sc in components]
        
    def __str__(self):
        """Default printer for SkyModel

        """
        s = "SkyModel:\n"
        for i, sc in enumerate(self.components):
            s += str(sc)
        s += "\n"
        
        for i, im in enumerate(self.images):
            s += str(im)
        s += "\n"
        
        return s