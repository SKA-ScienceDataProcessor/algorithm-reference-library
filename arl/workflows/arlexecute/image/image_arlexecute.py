
__all__ = ['image_arlexecute_map_workflow', 'sum_images_arlexecute']

import logging

from arl.processing_library.image import copy_image
from arl.wrappers.arlexecute.execution_support.arlexecute import arlexecute
from arl.processing_components.image import image_scatter_facets, image_gather_facets

log = logging.getLogger(__name__)

def image_arlexecute_map_workflow(im, imfunction, facets=1, overlap=0, taper=None, **kwargs):
    """Apply a function across an image: scattering to subimages, applying the function, and then gathering
    
    :param im: Image to be processed
    :param imfunction: Function to be applied
    :param facets: See image_scatter_facets
    :param overlap: image_scatter_facets
    :param taper: image_scatter_facets
    :param kwargs: kwargs for imfunction
    :return: output image
    """
    
    facets_list = arlexecute.execute(image_scatter_facets, nout=facets**2)(im, facets=facets, overlap=overlap,
                                                                    taper=taper)
    root_list = [arlexecute.execute(imfunction)(facet, **kwargs) for facet in facets_list]
    gathered = arlexecute.execute(image_gather_facets)(root_list, im, facets=facets, overlap=overlap,
                                                       taper=taper)
    return gathered


def sum_images_arlexecute(image_list, split=2):
    """ Sum a set of images

    :param image_list: List of (image, sum weights) tuples
    :param split: Split into
    :return: image
    """
    def sum_images(imagelist):
        out = copy_image(imagelist[0])
        out.data += imagelist[1].data
        return out
    
    if len(image_list) > split:
        centre = len(image_list) // split
        result = [sum_images_arlexecute(image_list[:centre])]
        result.append(sum_images_arlexecute(image_list[centre:]))
        return arlexecute.execute(sum_images, nout=2)(result)
    else:
        return arlexecute.execute(sum_images, nout=2)(image_list)
