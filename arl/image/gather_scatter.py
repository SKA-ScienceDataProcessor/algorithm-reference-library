
#
"""
Functions that define and manipulate images. Images are just data and a World Coordinate System.
"""

import logging
from typing import List

from arl.data.data_models import Image
from arl.image.iterators import   image_raster_iter, image_channel_iter
from arl.data.parameters import get_parameter

log = logging.getLogger(__name__)

def image_scatter_facets(im: Image, facets=1) -> List[Image]:
    """Scatter an image into a list of subimages using the  image_raster_iterator

    :param im: Image
    :param facets: Number of image partitions on each axis (2)
    :return: list of subimages
    """
    image_list = list()
    for facet in   image_raster_iter(im, facets=facets):
        image_list.append(facet)

    return image_list


def image_gather_facets(image_list: List[Image], im: Image, facets=1) -> Image:
    """Gather a list of subimages back into an image using the  image_raster_iterator

    :param image_list: List of subimages
    :param im: Output image
    :param facets: Number of image partitions on each axis (2)
    :return: list of subimages
    """
    for i, facet in enumerate(  image_raster_iter(im, facets=facets)):
        facet.data[...] = image_list[i].data[...]
    
    return im


def image_scatter_channels(im: Image, subimages=1) -> List[Image]:
    """Scatter an image into a list of subimages using the channels

    :param im: Image
    :param facets: Number of image partitions on each axis (2)
    :return: list of subimages
    """
    
    image_list = list()
    for slab in image_channel_iter(im, subimages=subimages):
        image_list.append(slab)
        
    assert len(image_list) == subimages, "Too many subimages scattered"
    
    return image_list


def image_gather_channels(image_list: List[Image], im: Image, subimages=1) -> Image:
    """Gather a list of subimages back into an image using the channel_iterator

    :param image_list: List of subimages
    :param im: Output image
    :param facets: Number of image partitions on each axis (2)
    :return: list of subimages
    """
    
    for i, slab in enumerate(image_channel_iter(im, subimages=subimages)):
        slab.data[...] = image_list[i].data[...]
    
    return im
