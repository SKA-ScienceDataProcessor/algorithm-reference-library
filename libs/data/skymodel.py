import logging

from libs.image.operations import copy_image
from libs.skycomponent.base import copy_skycomponent

log = logging.getLogger(__name__)

class SkyModel:
    """ A model for the sky
    """
    
    def __init__(self, images=[], components=[], fixed=False):
        """ A model of the sky as a list of images and a list of components
        
        """
        self.images = [copy_image(im) for im in images]
        self.components = [copy_skycomponent(sc) for sc in components]
        self.fixed = fixed
        
    def __str__(self):
        """Default printer for SkyModel

        """
        s = "SkyModel: fixed: %s\n" % self.fixed
        for i, sc in enumerate(self.components):
            s += str(sc)
        s += "\n"
        
        for i, im in enumerate(self.images):
            s += str(im)
        s += "\n"
        
        return s