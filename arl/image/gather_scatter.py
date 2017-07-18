
#
"""
Functions that define and manipulate images. Images are just data and a World Coordinate System.
"""

import logging
from typing import List

from arl.data.data_models import Image
from arl.image.iterators import raster_iter

log = logging.getLogger(__name__)

def image_scatter(im: Image, **kwargs) -> List[Image]:
    """Scatter an image into a list of subimages using the raster_iterator

    :param im: Image
    :param facets: Number of image partitions on each axis (2)
    :return: list of subimages
    """
    
    image_list = list()
    for facet in raster_iter(im, **kwargs):
        image_list.append(facet)

    return image_list


def image_gather(image_list: List[Image], im: Image, **kwargs) -> Image:
    """Gather a list of subimages back into an image using the raster_iterator

    :param image_list: List of subimages
    :param im: Output image
    :param facets: Number of image partitions on each axis (2)
    :return: list of subimages
    """
    
    for i, facet in enumerate(raster_iter(im, **kwargs)):
        facet.data[...] = image_list[i].data[...]
    
    return im